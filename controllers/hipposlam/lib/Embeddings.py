import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from umap.parametric_umap import load_ParametricUMAP
from .DataLoaders import EmbeddingImageDatasetAll


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


def measure_umap_similarity(unew, umat, umins, umaxs):
    '''

    Parameters
    ----------
    unew : ndarray
        Shape= (Embeds_dim, ). float32.
    umat : ndarray
        Shape= (N, Embeds_dim). float32.
    umins : ndarray
        Shape= (Embeds_dim, ). float32.
    umaxs : ndarray
        Shape= (Embeds_dim, ). float32.

    Returns
    -------
    maxid: int
        Index of the embedding from the umat which has the highest similarity to unew.

    sim_measure: np.ndarray
        Vector of similarity measures between umat and unew.
    '''

    unew_norm = (unew - umins) / (umaxs - umins)
    umat_norm = (umat - umins) / (umaxs - umins)
    sim_measure = 1 - np.sqrt(np.sum(np.square(unew_norm.reshape(1, -1) - umat_norm), axis=1)) / np.sqrt(2)

    maxid = np.argmax(sim_measure)
    return maxid, sim_measure


def get_dataloaders(load_annotation_pth, load_embed_dir):
    dataset = EmbeddingImageDatasetAll(load_annotation_pth, load_embed_dir)
    generator1 = torch.Generator().manual_seed(0)
    train_dataset, test_dataset = random_split(dataset, [8000, 2032], generator=generator1)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=True)
    return train_dataloader, test_dataloader, train_dataset, test_dataset


if __name__ == "__main__":
    pass
