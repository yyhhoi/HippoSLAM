import os

from matplotlib import pyplot as plt
from torchvision import models

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
    def __init__(self, kld_mul=0.01, lr=0.001, weight_decay=0, input_dim=576, hidden_dim=256, bottleneck_dim=50):
        self.vae = VAE(input_dim, hidden_dim, bottleneck_dim)
        self.vae_opt = torch.optim.Adam(params=self.vae.parameters(), lr=lr, weight_decay=weight_decay)
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

        y, mu, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(y, x)
        kld_loss = self.kld(mu, logvar)
        loss = recon_loss + self.kld_mul * kld_loss
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()

        return loss.item(), (recon_loss.item(), kld_loss.item())


    @staticmethod
    def kld(mu, logvar):
        out = 1 + logvar - mu.pow(2) - logvar.exp()
        return torch.mean(-0.5 * torch.sum(out, dim=1), dim=0)

    def save_checkpoint(self, pth):
        ckpt_dict = {
            'vae_state_dict': self.vae.state_dict(),
            'vae_opt_state_dict': self.vae_opt.state_dict(),
            'kld_mul': self.kld_mul,
        }
        torch.save(ckpt_dict, pth)

    def load_checkpoint(self, pth):
        checkpoint = torch.load(pth)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        self.vae_opt.load_state_dict(checkpoint['vae_opt_state_dict'])
        self.kld_mul = checkpoint['kld_mul']






def TrainVAE():
    data_dir = 'F:\VAE'
    # data_dir = 'data\VAE'
    load_embed_dir = join(data_dir, 'embeds')
    load_annotation_pth = join(data_dir, 'annotations.csv')
    save_plot_dir = join(data_dir, 'plots')
    os.makedirs(save_plot_dir, exist_ok=True)
    save_ckpt_pth = join(data_dir, 'ckpt.pt')


    # Prepare datasets
    dataset = EmbeddingImageDataset(load_annotation_pth, load_embed_dir)
    generator1 = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = random_split(dataset, [8000, 2003], generator=generator1)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # Model
    vaelearner = VAELearner(kld_mul=0.01,
                            lr=0.001,
                            weight_decay=0)

    # Training
    vaelearner.vae.train()
    num_epoches = 100
    loss_log = {'recon':[], 'kld':[]}
    for ei in range(num_epoches):
        print('\rTraining epoch %d/%d'%(ei, num_epoches), flush=True, end='')
        losstmp = {'recon':[], 'kld':[]}
        for x_train, _ in iter(train_dataloader):
            loss, (recon_loss, kld_loss) = vaelearner.train(x_train)



        loss_log['recon'].append(recon_loss)
        loss_log['kld'].append(kld_loss)
    vaelearner.save_checkpoint(save_ckpt_pth)

    # Plot loss
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(loss_log['recon'])
    ax[1].plot(loss_log['kld'])
    fig.savefig(join(save_plot_dir, 'Losses.png'), dpi=300)

if __name__ == "__main__":

    TrainVAE()