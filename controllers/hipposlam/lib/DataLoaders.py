import math
import os
from .circ import cdiff
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
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
    def __init__(self, load_annotation_pth, load_embed_dir):
        self.img_labels = pd.read_csv(load_annotation_pth, header=0)
        self.load_embed_dir = load_embed_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        embed_path = os.path.join(self.load_embed_dir, '%d.pt'%self.img_labels.iloc[idx, 0])
        embed = torch.load(embed_path)
        label = torch.tensor(self.img_labels.iloc[idx, [1, 2, 3]], dtype=torch.float32)
        return embed, label

class EmbeddingImageDatasetAll(EmbeddingImageDataset):
    def __init__(self, load_annotation_pth, load_embed_dir, to_numpy=False):
        super().__init__(load_annotation_pth, load_embed_dir)
        self.embed_all = torch.load(join(self.load_embed_dir, 'all.pt'))
        self.to_numpy = to_numpy

    def __getitem__(self, idx):
        embed = self.embed_all[idx, :]
        label = torch.tensor(self.img_labels.iloc[0][list('xya')], dtype=torch.float32)

        if self.to_numpy:
            return embed.numpy(), label.numpy()
        else:
            return embed, label

    def get_all(self):
        if self.to_numpy:
            return self.embed_all.numpy(), self.img_labels[['x', 'y', 'a']].to_numpy()
        else:
            return self.embed_all, torch.tensor(self.img_labels[list('xya')], dtype=torch.float32)



class ContrastiveEmbeddingDataloader:
    def __init__(self, load_annotation_pth, load_embed_dir, batchsize, datainds):
        self.load_annotation_pth = load_annotation_pth
        self.load_embed_dir = load_embed_dir
        self.batchsize = batchsize
        self.dist_thresh= 1
        self.adiff_thresh = np.pi/12
        self.embed_all = torch.load(join(self.load_embed_dir, 'all.pt'))[datainds[0]:datainds[1], ...]
        labels_df = pd.read_csv(load_annotation_pth, header=0)
        self.labels_xya = torch.from_numpy(labels_df[['x', 'y', 'a']].to_numpy())[datainds[0]:datainds[1], ...]
        assert self.embed_all.shape[0] == self.labels_xya.shape[0]


    def __len__(self):
        return self.embed_all.shape[0]

    def iterate(self):
        N = len(self)
        randvec = np.random.permutation(N)
        slice_inds = np.append(np.arange(0, N, self.batchsize), N)
        for i in range(len(slice_inds)-1):
            start_ind, end_ind = slice_inds[i], slice_inds[i+1]

            randinds = randvec[start_ind:end_ind]
            data = self.embed_all[randinds, :]
            labels = self.labels_xya[randinds, :]
            sim_mask, dissim_mask = self._compute_contrastive_masks(labels[:, 0], labels[:, 1], labels[:, 2])

            yield data, labels, sim_mask, dissim_mask


    def _compute_contrastive_masks(self, x, y, a):
        adiff = torch.abs(cdiff(a.reshape(-1, 1), a.reshape(1, -1)))
        posdist = torch.sqrt((x.reshape(-1, 1) - x.reshape(1, -1)) ** 2 + (
                    y.reshape(-1, 1) - y.reshape(1, -1)) ** 2) / math.sqrt(2)
        sim_mask_tmp = ((posdist < self.dist_thresh) & (adiff < self.adiff_thresh)).to(torch.uint8)  # (N, N)
        no_diag_mask = 1 - torch.eye(sim_mask_tmp.shape[0]).to(torch.uint8)
        sim_mask = sim_mask_tmp & no_diag_mask  # (N, N). dtype=torch.uint8
        dissim_mask = 1 - sim_mask_tmp  # (N, N). dtype=torch.uint8
        return sim_mask, dissim_mask


class LocalContrastiveEmbeddingDataloader:
    def __init__(self, load_annotation_pth, load_embed_dir, context_window, datainds):
        self.load_annotation_pth = load_annotation_pth
        self.load_embed_dir = load_embed_dir
        self.context_window = context_window
        self.embed_all = torch.load(join(self.load_embed_dir, 'all.pt'))[datainds[0]:datainds[1], ...]
        labels_df = pd.read_csv(load_annotation_pth, header=0).to_numpy()
        self.labels_xya = torch.from_numpy(labels_df)[datainds[0]:datainds[1], [2, 3, 4]]
        self.labels_t = torch.from_numpy(labels_df)[datainds[0]:datainds[1], 1].to(torch.int64)

    def __len__(self):
        return self.embed_all.shape[0]

    def iterate(self):

        inds = torch.where(torch.diff(self.labels_t) < 1)[0] + 1
        inds = torch.concat([torch.tensor([0]), inds, torch.tensor([self.labels_t.shape[0]])])

        for i in range(len(inds) - 1):
            start_ind, end_ind = inds[i], inds[i + 1]
            t = self.labels_t[start_ind:end_ind]
            data = self.embed_all[start_ind:end_ind, :]
            labels = self.labels_xya[start_ind:end_ind, :]
            dt = torch.abs(t.reshape(-1, 1) - t.reshape(1, -1))
            sim_mask = ((dt <= self.context_window) & (dt > 0)).to(torch.int64)
            no_diag_mask = 1 - torch.eye(sim_mask.shape[0]).to(torch.int64)
            dissim_mask = (1 - sim_mask) & no_diag_mask  # (N, N). dtype=torch.int64
            yield data, labels, sim_mask, dissim_mask

