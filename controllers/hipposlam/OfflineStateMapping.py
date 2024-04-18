import os.path

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import umap
from tqdm import tqdm
from hipposlam_lib.Embeddings import EmbeddingImageDatasetAll, VAELearner, convert_to_embed
import torch
from torch.utils.data import DataLoader
import pickle

class OfflineEmbeddingLearner:
    def __init__(self, lowSThresh):

        self.lowSThresh = lowSThresh
        self.sid2embed = []

    def observe(self, e_new, largest_eudist):
        # Initial condition
        if len(self.sid2embed) == 0:
            self.sid2embed.append(e_new.copy())
            return len(self.sid2embed) - 1, 0

        e_mat = np.stack(self.sid2embed)  # -> (Nstates, Embed_dim)

        # # Cosine similarity. (Embed_dim, ) @ (Nstates, Embed_dim).T -> (Nstates, )
        # cossim = e_new @ (e_mat.T) / (np.linalg.norm(e_new) * np.linalg.norm(e_mat, axis=-1) + 1e-9)
        # sim_measure = cossim
        sim_measure = self.prenorm_eudist(e_new, largest_eudist)
        maxid = np.argmax(sim_measure)
        maxcossim = sim_measure[maxid]

        if (maxcossim < self.lowSThresh):  # Not matching any existing embeddings
            # Create a new state, and remember the embedding
            self.sid2embed.append(e_new.copy())
            return len(self.sid2embed) - 1, 1

        else:
            return maxid, maxcossim


    def prenorm_eudist(self, e_new, e_mat, largest_eudist):
        eusim = 1 - np.sqrt(np.sum(np.square(e_new.reshape(1, -1) - e_mat), axis=1))/largest_eudist.reshape(1, -1)
        return eusim




# Load data
project_dir = join('data', 'VAE')
load_embed_dir = join(project_dir, 'embeds2')
load_annotation_pth = join(project_dir, 'annotations2.csv')
save_dir = join(project_dir, 'model', 'OnlyEmbed')
save_umap_pth = join(save_dir, 'umap.pickle')
dataset = EmbeddingImageDatasetAll(load_annotation_pth, load_embed_dir, to_numpy=True)
MNembeds = np.stack([e for e, _ in dataset])
labels = np.stack([e for _, e in dataset])


# Fit and transform umap
nneigh = 10
min_dist = 0.9
metric = 'euclidean'
if os.path.exists(save_umap_pth):
    with open(save_umap_pth, 'rb') as f:
        umap_model = pickle.load(f)
else:
    umap_model = umap.UMAP(n_neighbors=nneigh,
                           min_dist=min_dist,
                           metric=metric,
                           n_components=2)
with open(save_umap_pth, 'wb') as f:
    pickle.dump(umap_model, f)
umap_embeds = umap_model.fit_transform(MNembeds)



# Plot Umap
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax = ax.ravel()
im0 = ax[0].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=labels[:, 0], cmap='jet', alpha=0.5)
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=labels[:, 1], cmap='jet', alpha=0.5)
plt.colorbar(im1, ax=ax[1])
im2 = ax[2].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=labels[:, 2], cmap='hsv', alpha=0.5)
plt.colorbar(im2, ax=ax[2])
txt = f'nneigh={nneigh}\nmindist={min_dist:0.2f}\nmetric={metric}\n'
ax[3].annotate(text=txt, xy=(0.2, 0.5), xycoords='axes fraction')
ax[3].axis('off')
fig.tight_layout()
fig.savefig(join(save_dir, 'UmapEmbeddingCheck.png'), dpi=300)
plt.close(fig)


# Normalize umap embeddings

umap_embeds.max()

# Generate states



lowSThresh = 0.8
oel = OfflineEmbeddingLearner(lowSThresh=lowSThresh)
Niters = 10000
sid = np.zeros(Niters)
sval = np.zeros(Niters)
obsnum = np.zeros(Niters)
for i in tqdm(range(Niters)):
    sid[i], sval[i] = oel.observe(umap_embeds[i, :])
    obsnum[i] = len(oel.sid2embed)
