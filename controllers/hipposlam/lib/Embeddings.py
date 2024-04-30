import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
import numpy as np
import torch
from umap.parametric_umap import load_ParametricUMAP


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
    Compare how similar the new embedding vector (unew) is to all previous old embedding vectors (umat), normalized
    by the max and min of the whole embedding space (umins and umaxs).

    Parameters
    ----------
    unew : ndarray
        Shape= (Embeds_dim, ). float32. New embedding vector.
    umat : ndarray
        Shape= (N, Embeds_dim). float32. Storage of Umap embeddings.
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


if __name__ == "__main__":
    pass
