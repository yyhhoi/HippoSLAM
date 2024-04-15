import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
from umap.parametric_umap import ParametricUMAP
from tqdm import tqdm
from .Embeddings import save_parametric_umap_model, load_parametric_umap_model
from .utils import read_pickle, save_pickle
from .vision import MobileNetEmbedder

from .Sequences import Sequences, StateDecoder

def ImageSampling(assets_dir):
    from .Environments import StateMapLearnerImageSaver
    save_hipposlam_pth = join(assets_dir, 'hipposlam.pickle')
    save_trajdata_pth = join(assets_dir, 'trajdata.pickle')
    save_img_dir = join(assets_dir, 'imgs')
    os.makedirs(save_img_dir, exist_ok=True)

    env = StateMapLearnerImageSaver(R=5, L=20, maxt=1000, max_hipposlam_states=500,
                 save_hipposlam_pth=save_hipposlam_pth, save_trajdata_pth=save_trajdata_pth, save_img_dir=save_img_dir)

    max_img_num = 10000
    env.c = 0
    while env.c < max_img_num:
        print(r'%d/%d'%(env.c, max_img_num), end='', flush=True)
        env.reset()

        terminated = False
        truncated = False

        while (not terminated) and (not truncated):
            s, reward, terminated, truncated, info = env.step(np.random.choice(4))

    env.reset()

def preprocess_trajdata(assets_dir, load_trajdata_pth):
    """
    Concatenate a list of episode data and generate a dataframe.
    Load:
        trajdata.pickle
    Save:
        trajdf.pickle
    """

    keys = ['x', 'y', 'a', 't', 'fsigma', 'c']
    data_dict = {key: [] for key in keys}
    trajdata = read_pickle(load_trajdata_pth)  # list of dicts. The key in the dict contains a list of data.
    for i in range(len(trajdata)):
        episode = trajdata[i]
        for key in keys:
            data_dict[key].extend(episode[key])
    trajdf = pd.DataFrame(data_dict)

    # Add column: img_name c_t
    f = lambda x: str(x['c']) + '_' + str(x['t']) + '.png'
    trajdf['img_name'] = trajdf.apply(f, axis=1)
    trajdf['img_exist'] = trajdf['t'] % 5 == 0
    trajdf.to_pickle(join(assets_dir, 'trajdf.pickle'))
    return trajdf

def convert_images_to_mobilenet_embeddings(assets_dir, load_trajdf_pth, load_img_dir):
    """
    Load:
        trajdf.pickle
        imgs/*.png
    Save:
        embeds_index.pickle
        mobilenet_embeds.pt
        annotations.csv
    """

    # Load embeddings
    trajdf = read_pickle(load_trajdf_pth)
    img_mask = trajdf['img_exist']
    subdf = trajdf[img_mask]
    img_name_list = subdf['img_name'].tolist()

    # Save annotations and indexes
    annotations = subdf[['x', 'y', 'a']].reset_index(drop=True)
    annotations.to_csv(join(assets_dir, 'annotations.csv'))
    embed_index = {subdf.index[i]: i for i in range(len(subdf.index))}
    save_pickle(join(assets_dir, 'embeds_index.pickle'), embed_index)

    # Infer embeddings
    num_imgs = len(img_name_list)
    embeds = torch.zeros((num_imgs, 576))
    embedder = MobileNetEmbedder()
    for i, img_name in enumerate(tqdm(img_name_list)):
        load_img_pth = join(load_img_dir, img_name)
        embeds[i, :] = embedder.infer_embedding_from_path(load_img_pth).squeeze()  # -> (576, )

    # Save embeddings
    torch.save(embeds, join(assets_dir, 'mobilenet_embeds.pt'))
    return embeds


def convert_embeddings_mobilenet_to_umap(load_embeds_pth, load_annotations_pth, save_umap_dir):
    """
    Load:
        mobilenet_embeds.pt
        annotations.csv
    Save:
        # Defined in this code block
        save_umap_dir/umap_training_log.png
        save_umap_dir/umap_embeddings.pt
        save_umap_dir/umap_embeds_vis.png
        save_umap_dir/umins.pt
        save_umap_dir/umaxs.pt

        # Defined automatically by umap libraries
        save_umap_dir/encoder.keras
        save_umap_dir/model.pkl
        save_umap_dir/parametric_model.keras
    """

    # Load embeds
    mobilenet_embeds = torch.load(load_embeds_pth)

    # Fit and transform umap
    nneigh = 10
    min_dist = 0.9
    metric = 'euclidean'
    umap_model = ParametricUMAP(n_neighbors=nneigh,
                                min_dist=min_dist,
                                metric=metric,
                                n_components=2)
    umap_embeds = umap_model.fit_transform(mobilenet_embeds)


    # Plot the training loss
    fig, ax = plt.subplots()
    ax.plot(umap_model._history['loss'])
    ax.set_ylabel('Cross Entropy')
    ax.set_xlabel('Epoch')
    fig.savefig(join(save_umap_dir, 'umap_training_log.png'))


    # Get max and min for normalization
    umins = umap_embeds.min(axis=0)
    umaxs = umap_embeds.max(axis=0)
    print('Umap embeds mins\n', umins)
    print('Umap embeds maxs\n', umaxs)

    # Save Umap
    save_parametric_umap_model(save_umap_dir, umins, umaxs)
    torch.save(umap_embeds, join(save_umap_dir, 'umap_embeddings.pt'))

    # Load Annotation
    ann_df = pd.read_csv(load_annotations_pth)

    # Plot Umap embedding space and save
    fig, ax = plot_umap_embeddings_2d(umap_embeds, ann_df, nneigh, min_dist, metric)
    fig.savefig(join(save_umap_dir, 'umap_embeds_vis.png'), dpi=300)


def check_trained_umap_model(load_embeds_pth, load_annotations_pth, load_umap_dir):
    """
    Load:
        mobilenet_embeds.pt
        annotations.csv
        save_umap_dir/* (except umap_embeddings.pt and *.png)
    Save:
        save_umap_dir/umap_embeds_vis_check.png

    """

    # Load mobilenet embeds
    mobilenet_embeds = torch.load(load_embeds_pth)

    # Load Annotation
    ann_df = pd.read_csv(load_annotations_pth)

    # Load umap model (previously trained)
    umap_model, umins, umaxs = load_parametric_umap_model(load_umap_dir)

    # Inference and check
    umap_embeds = umap_model.transform(mobilenet_embeds)
    fig, ax = plot_umap_embeddings_2d(umap_embeds, ann_df, None, None, None)
    fig.savefig(join(load_umap_dir, 'umap_embeds_vis_check.png'), dpi=300)


def plot_umap_embeddings_2d(umap_embeds, ann_df, nneigh, min_dist, metric):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax = ax.ravel()
    im0 = ax[0].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['x'], cmap='jet', alpha=0.5)
    plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['y'], cmap='jet', alpha=0.5)
    plt.colorbar(im1, ax=ax[1])
    im2 = ax[2].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['a'], cmap='hsv', alpha=0.5)
    plt.colorbar(im2, ax=ax[2])
    txt = f'nneigh={nneigh}\nmindist={min_dist:0.2f}\nmetric={metric}\n' if nneigh is not None else ''
    ax[3].annotate(text=txt, xy=(0.2, 0.5), xycoords='axes fraction')
    ax[3].axis('off')
    fig.tight_layout()
    return fig, ax

def statemap_learn(load_trajdf_pth, load_embedsIndex_pth, load_umapEmbeds_pth, load_umap_dir):
    """
    Load:
        trajdf.pickle
        embeds_index.pickle
        umap_embeddings.pt
        load_umap_dir/umins.pt
        load_umap_dir/umaxs.pt



    """
    # Parameters
    R, L = 5, 20

    # Load data
    trajdf = read_pickle(load_trajdf_pth)
    embeds_index = read_pickle(load_embedsIndex_pth)
    umap_embeddings = torch.load(load_umapEmbeds_pth).numpy()
    umins = torch.load(join(load_umap_dir, 'umins.pt')).numpy()
    umaxs = torch.load(join(load_umap_dir, 'umaxs.pt')).numpy()


    # Initialize hipposlam
    hipposeq = Sequences(R=R, L=L, reobserve=False)
    hippomap = StateDecoder(R=R, L=L, maxN=1000, area_norm=False)

    for i in tqdm(range(trajdf.shape[0])):
        id_list = trajdf.loc[i, 'id_list']

        hipposeq.step(id_list)
        sid, Snodes = hippomap.infer_state(hipposeq.X)

        if (i in embeds_index) and (hippomap.current_F > 0):
            umap_embed = umap_embeddings[embeds_index[i], :]  # -> (2, )

            current_embedid = hippomap.learn_embedding(hipposeq.X, umap_embed, umins, umaxs,
                                                                 far_ids=None)

