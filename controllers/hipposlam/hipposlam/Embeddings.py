import os
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models
from tqdm import tqdm
from .utils import Recorder
from .Networks import VAE
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

from torchvision.io import read_image
from .DataLoaders import WebotsImageDataset, EmbeddingImageDataset, EmbeddingImageDatasetAll, \
    LocalContrastiveEmbeddingDataloader, ContrastiveEmbeddingDataloader
import logging
from glob import glob

def save_parametric_umap_model(umap_model, save_umap_dir, umins, umaxs):
    umap_model.save(save_umap_dir)
    torch.save(umins, join(save_umap_dir, 'umins.pt'))
    torch.save(umaxs, join(save_umap_dir, 'umaxs.pt'))

def load_parametric_umap_model(load_umap_dir):
    # Load umap model (previously trained)
    umap_model = load_ParametricUMAP(load_umap_dir)
    umins = torch.load(join(load_umap_dir, 'umins.pt'))
    umaxs = torch.load(join(load_umap_dir, 'umaxs.pt'))
    return umap_model, umins, umaxs

def convert_to_embed(load_img_dir, load_annotation_pth, save_embed_dir, all=True):
    """
    Run MobileNet V3 Small to convert images to embeddings, and save.

    """

    os.makedirs(save_embed_dir, exist_ok=True)

    dataset = WebotsImageDataset(load_annotation_pth, load_img_dir)

    # Load MobileNet
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    all_embeds = []
    for i in range(len(dataset)):
        save_embed_pth = join(save_embed_dir, f'{i}.pt')

        if os.path.exists(save_embed_pth):
            continue
        print('\r%d/%d' % (i, len(dataset)), flush=True, end='')
        image, _ = dataset[i]
        batch_t = preprocess(image).unsqueeze(0)
        # Get the features from the model

        with torch.no_grad():
            x = model.features(batch_t)
            x = model.avgpool(x)
            embedding = torch.flatten(x)
            if all:
                all_embeds.append(embedding)
            else:
                torch.save(embedding, save_embed_pth)
    if all:
        torch.save(torch.stack(all_embeds), join(save_embed_dir, 'all.pt'))

def combine_embeds(load_embeds_dir):
    pths = glob(join(load_embeds_dir, '*'))
    all_embeds = []
    for pth in pths:
        embed = torch.load(pth)
        all_embeds.append(embed)
    torch.save(torch.stack(all_embeds), join(load_embeds_dir, 'all.pt'))


class VAELearner:
    def __init__(self, input_dim, hidden_dims, lr=0.001, lr_gamma=0.9, weight_decay=0):
        self.vae = VAE(input_dim, hidden_dims)  # input_dim = 576
        self.vae_opt = torch.optim.Adam(params=self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_opt = ExponentialLR(self.vae_opt, gamma=lr_gamma)

    def train(self, x, kld_mul):
        """

        Parameters
        ----------
        x : tensor
            (batch_size, input_dim) torch.flaot32. Image Embeddings.

        Returns
        -------
        loss, (recon_loss, kld_loss)

        """

        loss, (recon_loss, kld_loss), _ = self._forward_pass(x, kld_mul)
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()
        return loss.item(), (recon_loss.item(), kld_loss.item())

    def infer(self, x, kld_mul):
        with torch.no_grad():
            loss, (recon_loss, kld_loss), (y, mu, logvar) = self._forward_pass(x, kld_mul)
        return loss.item(), (recon_loss.item(), kld_loss.item()), (y.detach(), mu.detach(), logvar.detach())


    def _forward_pass(self, x, kld_mul):
        y, mu, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(y, x)
        kld_loss = self.kld(mu, logvar)
        loss = recon_loss + kld_mul * kld_loss
        return loss, (recon_loss, kld_loss), (y, mu, logvar)


    @staticmethod
    def kld(mu, logvar):
        out = 1 + logvar - mu.pow(2) - logvar.exp()
        return torch.mean(-0.5 * torch.sum(out, dim=1), dim=0)

    def save_checkpoint(self, pth):
        ckpt_dict = {
            'vae_state_dict': self.vae.state_dict(),
            'vae_opt_state_dict': self.vae_opt.state_dict(),
            'lr_opt': self.lr_opt.state_dict(),
        }
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae_opt.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.lr_opt.load_state_dict(checkpoint['lr_opt'])


class ContrastiveVAELearner(VAELearner):
    def __init__(self, input_dim, hidden_dims, con_margin=0.1, con_mul=0.1, dismul=1.0, lr=0.001, lr_gamma=0.9, weight_decay=0):
        super().__init__(input_dim, hidden_dims, lr, lr_gamma, weight_decay)
        self.con_mul = con_mul
        self.con_margin = con_margin
        self.dismul = dismul

    def contrastive_forward(self, x, kld_mul, sim_mask, dissim_mask):
        y, mu, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(y, x)
        kld_loss = self.kld(mu, logvar)
        cdist = torch.cdist(mu, mu)  # (N, bottleneck_dim) -> (N, N)
        con_loss = torch.mean(sim_mask * cdist + self.dismul * dissim_mask * nn.functional.relu(self.con_margin - cdist))

        loss = recon_loss + kld_mul * kld_loss + self.con_mul * con_loss
        return loss, (recon_loss, kld_loss, con_loss)

    def train_contrastive(self, x, kld_mul, sim_mask, dissim_mask):
        loss, (recon_loss, kld_loss, con_loss) = self.contrastive_forward(x, kld_mul, sim_mask, dissim_mask)

        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()
        return loss.item(), (recon_loss.item(), kld_loss.item(), con_loss.item())

    def test_contrastive(self, x, kld_mul, sim_mask, dissim_mask):
        with torch.no_grad():
            loss, (recon_loss, kld_loss, con_loss) = self.contrastive_forward(x, kld_mul, sim_mask, dissim_mask)
        return loss.item(), (recon_loss.item(), kld_loss.item(), con_loss.item())


    def save_checkpoint(self, pth):
        ckpt_dict = {
            'vae_state_dict': self.vae.state_dict(),
            'vae_opt_state_dict': self.vae_opt.state_dict(),
            'lr_opt': self.lr_opt.state_dict(),
            'con_mul': self.con_mul,
            'con_margin': self.con_margin,
        }
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae_opt.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.lr_opt.load_state_dict(checkpoint['lr_opt'])
        self.con_mul = checkpoint['con_mul']
        self.con_margin = checkpoint['con_margin']

def get_dataloaders(load_annotation_pth, load_embed_dir):

    dataset = EmbeddingImageDatasetAll(load_annotation_pth, load_embed_dir)
    generator1 = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = random_split(dataset, [8000, 2032], generator=generator1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def TrainVAE(kld_mul=1):

    model_tag = f'Annealing_bottle25'
    data_dir = join('data', 'VAE')
    load_embed_dir = join(data_dir, 'embeds2')
    load_annotation_pth = join(data_dir, 'annotations2.csv')
    save_dir = join(data_dir, 'model', model_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare datasets
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloaders(load_annotation_pth, load_embed_dir)

    # Model & Set-up
    vaelearner = VAELearner(
        input_dim=576,
        hidden_dims=[400, 200, 100, 50, 25],
        kld_mul=kld_mul,
        lr=0.001,
        lr_gamma=0.98,
        weight_decay=0
    )

    cycle_nums = 10
    num_epoches = 21
    betas = np.linspace(0, 1, num_epoches)

    keys = [f'{a}_{b}' for a in ['recon', 'kld'] for b in ['train', 'test']]
    loss_recorder = Recorder(*(keys + ['lr', 'beta']))

    for cyci in range(cycle_nums):
        cyclemodel_tag = f'{model_tag}_cycle{cyci}'
        save_ckpt_pth = join(save_dir, f'ckpt_{cyclemodel_tag}.pt')
        for ei in tqdm(range(num_epoches)):

            print('\rTraining epoch %d/%d'%(ei, num_epoches), flush=True, end='')
            epoch_recorder = Recorder(*keys)

            # Training
            vaelearner.vae.train()
            for x_train, _ in iter(train_dataloader):
                loss_train, (recon_loss_train, kld_loss_train) = vaelearner.train(x_train, betas[ei])
                epoch_recorder.record(recon_train=recon_loss_train, kld_train=kld_loss_train)

            # Testing
            vaelearner.vae.eval()
            for x_test, _ in iter(test_dataloader):
                loss_test, (recon_loss_test, kld_loss_test), _ = vaelearner.infer(x_test, betas[ei])
                epoch_recorder.record(recon_test=recon_loss_test, kld_test=kld_loss_test)
            avers_dict = epoch_recorder.return_avers()
            loss_recorder.record(**avers_dict)
            loss_recorder.record(lr=vaelearner.lr_opt.get_last_lr()[0], beta=betas[ei])
            # vaelearner.lr_opt.step()
        vaelearner.save_checkpoint(save_ckpt_pth)

    # Plot loss
    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(loss_recorder['recon_train'], label='recon_train')
    ax[0].plot(loss_recorder['recon_test'], label='recon_test')
    ax[0].set_title('train=%0.6f, test=%0.6f'% (loss_recorder['recon_train'][-1], loss_recorder['recon_test'][-1]))
    ax[0].legend()
    ax[1].plot(loss_recorder['kld_train'], label='kld_train')
    ax[1].plot(loss_recorder['kld_test'], label='kld_test')
    ax[1].set_title('train=%0.6f, test=%0.6f' % (loss_recorder['kld_train'][-1], loss_recorder['kld_test'][-1]))
    ax[1].legend()
    for axeach in ax[:2]:
        axeach.set_ylim(0, 0.1)
    ax[2].plot(loss_recorder['lr'], marker='x')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Learning rate')
    ax[3].plot(loss_recorder['beta'], marker='x')
    fig.savefig(join(save_dir, f'Losses_{model_tag}.png'), dpi=300)
    plt.close(fig)
    loss_recorder.to_csv(join(save_dir, 'Loss.csv'))


    # Plot examples
    vaelearner.vae.eval()
    x_test, _ = next(iter(test_dataloader))
    nexamples = 2
    fig, ax = plt.subplots(nexamples, 1, figsize=(16, 8), sharex=True)
    x_ax = np.arange(x_test.shape[1])
    for i in range(nexamples):
        xtest, _ = test_dataset[i]
        loss_test, (recon_loss_test, kld_loss_test), (y, mu, logvar) = vaelearner.infer(xtest.unsqueeze(0), 1)

        ax[i].step(x_ax, x_test[0, :], color='r', label='input', linewidth=1)
        ax[i].step(x_ax, y[0, :], color='b', label='output', linewidth=1)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(join(save_dir, f'pred_examples_{model_tag}.png'), dpi=300)
    plt.close(fig)


def TrainContrastiveVAE(con_mul=0.1):

    data_dir = join('data', 'VAE')
    load_embed_dir = join(data_dir, 'embeds')
    load_annotation_pth = join(data_dir, 'annotations.csv')
    model_tag = f'ContrastiveVAEmul=%0.4f' %con_mul
    save_dir = join(data_dir, 'model', model_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare datasets
    train_dataloader = ContrastiveEmbeddingDataloader(load_annotation_pth, load_embed_dir, 256, [0, 8000])
    test_dataloader = ContrastiveEmbeddingDataloader(load_annotation_pth, load_embed_dir, 256, [8000, 10032])


    # Model & Set-up
    vaelearner = ContrastiveVAELearner(
        input_dim= 576,
        hidden_dims= [400, 200, 100, 50, 25],
        con_margin= 0.1,
        con_mul= con_mul,
        dismul = 0.5,
        lr = 0.001,
        lr_gamma=0.98,
        weight_decay=0
    )

    cycle_nums = 10
    num_epoches = 50
    betas = np.linspace(0, 1, num_epoches)

    keys = [f'{a}_{b}' for a in ['recon', 'kld', 'con'] for b in ['train', 'test']]
    loss_recorder = Recorder(*(keys + ['lr', 'beta']))

    for cyci in range(cycle_nums):
        cyclemodel_tag = f'{model_tag}_cycle{cyci}'
        save_ckpt_pth = join(save_dir, f'ckpt_{cyclemodel_tag}.pt')
        for ei in range(num_epoches):

            logging.info('Training epoch %d/%d'%(ei, num_epoches))
            epoch_recorder = Recorder(*keys)

            # Training
            vaelearner.vae.train()
            for x, _, sim_mask, dissim_mask in tqdm(train_dataloader.iterate()):
                if x.shape[0] < 2:
                    continue
                loss_train, (recon_loss_train, kld_loss_train, con_loss_train) = vaelearner.train_contrastive(
                    x, betas[ei], sim_mask, dissim_mask)
                logging.debug(np.around([loss_train, recon_loss_train, kld_loss_train, con_loss_train], 3))
                epoch_recorder.record(recon_train=recon_loss_train, kld_train=kld_loss_train, con_train=con_loss_train)

            # Testing
            vaelearner.vae.eval()
            with torch.no_grad():
                for x, _, sim_mask, dissim_mask in tqdm(test_dataloader.iterate()):
                    if x.shape[0] < 2:
                        continue
                    loss_test, (recon_loss_test, kld_loss_test, con_loss_test) = vaelearner.test_contrastive(
                        x, betas[ei], sim_mask, dissim_mask)
                    logging.debug(np.around([loss_test, recon_loss_test, kld_loss_test, con_loss_test], 3))
                    epoch_recorder.record(recon_test=recon_loss_test, kld_test=kld_loss_test,
                                          con_test=con_loss_test)


            avers_dict = epoch_recorder.return_avers()
            # for key, item in avers_dict.items():
            #     print(f'{key}: {item: 0.4f}')
            loss_recorder.record(**avers_dict)
            loss_recorder.record(lr=vaelearner.lr_opt.get_last_lr()[0], beta=betas[ei])
            # vaelearner.lr_opt.step()
        vaelearner.save_checkpoint(save_ckpt_pth)

    # Plot loss
    fig, ax = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    ax[0].plot(loss_recorder['recon_train'], label='recon_train')
    ax[0].plot(loss_recorder['recon_test'], label='recon_test')
    ax[0].set_title('train=%0.6f, test=%0.6f'% (loss_recorder['recon_train'][-1], loss_recorder['recon_test'][-1]))
    ax[0].legend()
    ax[1].plot(loss_recorder['kld_train'], label='kld_train')
    ax[1].plot(loss_recorder['kld_test'], label='kld_test')
    ax[1].set_title('train=%0.6f, test=%0.6f' % (loss_recorder['kld_train'][-1], loss_recorder['kld_test'][-1]))
    ax[1].legend()
    ax[2].plot(loss_recorder['con_train'], label='con_train')
    ax[2].plot(loss_recorder['con_test'], label='con_test')
    ax[2].set_title('train=%0.6f, test=%0.6f' % (loss_recorder['con_train'][-1], loss_recorder['con_test'][-1]))
    ax[2].legend()


    for axeach in ax[:3]:
        axeach.set_ylim(0, 0.1)
    ax[3].plot(loss_recorder['lr'], marker='x')
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Learning rate')
    ax[4].plot(loss_recorder['beta'], marker='x')
    fig.savefig(join(save_dir, f'Losses_{model_tag}.png'), dpi=300)
    plt.close(fig)
    loss_recorder.to_csv(join(save_dir, 'Loss.csv'))



if __name__ == "__main__":

    # # Convert Webots Camera images to MobileNetv3 embedding
    # data_dir = 'VAE'
    # load_img_dir = join(data_dir, 'imgs2')
    # load_annotation_pth = join(data_dir, 'annotations2.csv')
    # save_embed_dir = join(data_dir, 'embeds2')
    # convert_to_embed(load_img_dir, load_annotation_pth, save_embed_dir)
    # TrainVAE()
    TrainContrastiveVAE(0.1)
    # for kld_mul in [0.1, 0.01, 0.001]:
    #     print(f'\n {kld_mul:0.6f}')
    #     TrainVAE(kld_mul)
    #     print()
    # infer_examples()
