import os

from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import models
from tqdm import tqdm

from hipposlam.utils import Recorder
from hipposlam.Networks import VAE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
from glob import glob
import pickle
import numpy as np

import pandas as pd
import torch
import torch.nn as nn
from skimage.io import imsave
from torch.utils.data import Dataset, DataLoader, random_split


from torchvision.io import read_image


class WebotsImageDataset(Dataset):
    def __init__(self, load_annotation_pth, load_img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(load_annotation_pth, header=0)
        self.load_img_dir = load_img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        img_path = os.path.join(self.load_img_dir, '%d.png'%self.img_labels.iloc[idx, 0])
        image = read_image(img_path)[:3, ...]
        label = torch.tensor(self.img_labels.iloc[idx, [1, 2, 3]], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class EmbeddingImageDataset(Dataset):

    def __init__(self, load_annotation_pth, load_embed_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(load_annotation_pth, header=0)
        self.load_embed_dir = load_embed_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        embed_path = os.path.join(self.load_embed_dir, '%d.pt'%self.img_labels.iloc[idx, 0])
        embed = torch.load(embed_path)
        label = torch.tensor(self.img_labels.iloc[idx, [1, 2, 3]], dtype=torch.float32)

        if self.transform:
            embed = self.transform(embed)
        if self.target_transform:
            label = self.target_transform(label)
        return embed, label


def convert_to_embed():
    """
    Run MobileNet V3 Small to convert images to embeddings, and save.

    """
    data_dir = 'F:\VAE'
    load_img_dir = join(data_dir, 'imgs')
    load_annotation_pth = join(data_dir, 'annotations.csv')
    save_embed_dir = join(data_dir, 'embeds')
    os.makedirs(save_embed_dir, exist_ok=True)

    dataset = WebotsImageDataset(load_annotation_pth, load_img_dir)

    # Load MobileNet
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    for i in range(5000, len(dataset)):
        print('\r%d/%d'%(i, len(dataset)), flush=True, end='')
        image, _ = dataset[i]
        batch_t = preprocess(image).unsqueeze(0)
        # Get the features from the model

        with torch.no_grad():
            x = model.features(batch_t)
            x = model.avgpool(x)
            embedding = torch.flatten(x)
            torch.save(embedding, join(save_embed_dir, f'{i}.pt'))


class VAELearner:
    def __init__(self, input_dim, hidden_dims, kld_mul=0.01, lr=0.001, lr_gamma=0.9, weight_decay=0):
        self.vae = VAE(input_dim, hidden_dims)  # input_dim = 576
        self.vae_opt = torch.optim.Adam(params=self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr_opt = ExponentialLR(self.vae_opt, gamma=lr_gamma)
        self.kld_mul = kld_mul

    def train(self, x):
        """

        Parameters
        ----------
        x : tensor
            (batch_size, input_dim) torch.flaot32. Image Embeddings.

        Returns
        -------
        loss, (recon_loss, kld_loss)

        """

        loss, (recon_loss, kld_loss), _ = self._forward_pass(x)
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()
        return loss.item(), (recon_loss.item(), kld_loss.item())

    def infer(self, x):
        with torch.no_grad():
            loss, (recon_loss, kld_loss), (y, mu, logvar) = self._forward_pass(x)
        return loss.item(), (recon_loss.item(), kld_loss.item()), (y.detach(), mu.detach(), logvar.detach())


    def _forward_pass(self, x):
        y, mu, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(y, x)
        kld_loss = self.kld(mu, logvar)
        loss = recon_loss + self.kld_mul * kld_loss
        return loss, (recon_loss, kld_loss), (y, mu, logvar)



    @staticmethod
    def kld(mu, logvar):
        out = 1 + logvar - mu.pow(2) - logvar.exp()
        return torch.mean(-0.5 * torch.sum(out, dim=1), dim=0)

    def save_checkpoint(self, pth):
        ckpt_dict = {
            'vae_state_dict': self.vae.state_dict(),
            'vae_opt_state_dict': self.vae_opt.state_dict(),
            'kld_mul': self.kld_mul,
            'lr_opt': self.lr_opt.state_dict(),
        }
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae_opt.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.lr_opt.load_state_dict(checkpoint['lr_opt'])
        self.kld_mul = checkpoint['kld_mul']



def get_dataloaders(load_annotation_pth, load_embed_dir):

    dataset = EmbeddingImageDataset(load_annotation_pth, load_embed_dir)
    generator1 = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = random_split(dataset, [8000, 2003], generator=generator1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def TrainVAE(kld_mul=0.01):

    model_tag = f'kldmul={kld_mul:0.6f}_bottle50'
    data_dir = join('data', 'VAE')
    load_embed_dir = join(data_dir, 'embeds')
    load_annotation_pth = join(data_dir, 'annotations.csv')

    save_dir = join(data_dir, 'model', model_tag)
    os.makedirs(save_dir, exist_ok=True)
    save_ckpt_pth = join(save_dir, 'ckpt_%s.pt'%model_tag)

    # Prepare datasets
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloaders(load_annotation_pth, load_embed_dir)

    # Model
    vaelearner = VAELearner(
        input_dim=576,
        hidden_dims=[400, 200, 100, 50],
        kld_mul=kld_mul,
        lr=0.001,
        lr_gamma=0.95,
        weight_decay=0
    )

    # Training
    num_epoches = 20
    keys = [f'{a}_{b}' for a in ['recon', 'kld'] for b in ['train', 'test']]
    loss_recorder = Recorder(*(keys + ['lr']))
    for ei in tqdm(range(num_epoches)):
        print('\rTraining epoch %d/%d'%(ei, num_epoches), flush=True, end='')
        epoch_recorder = Recorder(*keys)

        # Training
        vaelearner.vae.train()
        for x_train, _ in iter(train_dataloader):
            loss_train, (recon_loss_train, kld_loss_train) = vaelearner.train(x_train)
            epoch_recorder.record(recon_train=recon_loss_train, kld_train=kld_loss_train)

        # Testing
        vaelearner.vae.eval()
        for x_test, _ in iter(test_dataloader):
            loss_test, (recon_loss_test, kld_loss_test), _ = vaelearner.infer(x_test)
            epoch_recorder.record(recon_test=recon_loss_test, kld_test=kld_loss_test)
        avers_dict = epoch_recorder.return_avers()
        loss_recorder.record(**avers_dict)
        loss_recorder.record(lr=vaelearner.lr_opt.get_last_lr()[0])
        vaelearner.lr_opt.step()
    vaelearner.save_checkpoint(save_ckpt_pth)

    # Plot loss
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(loss_recorder['recon_train'], label='recon_train')
    ax[0].plot(loss_recorder['recon_test'], label='recon_test')
    ax[0].set_title('train=%0.6f, test=%0.6f'% (loss_recorder['recon_train'][-1], loss_recorder['recon_test'][-1]))
    ax[0].legend()
    ax[1].plot(loss_recorder['kld_train'], label='kld_train')
    ax[1].plot(loss_recorder['kld_test'], label='kld_test')
    ax[1].set_title('train=%0.6f, test=%0.6f' % (loss_recorder['kld_train'][-1], loss_recorder['kld_test'][-1]))
    ax[1].legend()
    for axeach in ax[:2]:
        axeach.set_ylim(0, None)
    ax[2].plot(loss_recorder['lr'], marker='x')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Learning rate')
    fig.savefig(join(save_dir, f'Losses_{model_tag}.png'), dpi=300)
    plt.close(fig)

    # Plot examples
    vaelearner.vae.eval()


    x_test, _ = next(iter(test_dataloader))
    nexamples = 2
    fig, ax = plt.subplots(nexamples, 1, figsize=(16, 8), sharex=True)

    x_ax = np.arange(x_test.shape[1])
    for i in range(nexamples):
        xtest, _ = test_dataset[i]
        loss_test, (recon_loss_test, kld_loss_test), (y, mu, logvar) = vaelearner.infer(xtest.unsqueeze(0))

        ax[i].step(x_ax, x_test[0, :], color='r', label='input', linewidth=1)
        ax[i].step(x_ax, y[0, :], color='b', label='output', linewidth=1)

    ax[0].legend()
    fig.tight_layout()
    fig.savefig(join(save_dir, f'pred_examples_{model_tag}.png'), dpi=300)
    plt.close(fig)





if __name__ == "__main__":

    for kld_mul in [0.1, 0.01, 0.001]:
        print(f'\n {kld_mul:0.6f}')
        TrainVAE(kld_mul)
        print()
    # infer_examples()
