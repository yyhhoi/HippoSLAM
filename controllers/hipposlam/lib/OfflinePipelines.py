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
from .visualization import plot_spatial_specificity, compare_spatial_specificity
from .vision import MobileNetEmbedder

from .Sequences import Sequences, StateDecoder

def ImageSampling(run_dir):
    # The StateMapLearnerImageSaver is imported here as it would require the Webots libraries, which cannot be
    # conveniently imported in the terminal.
    from .Environments import StateMapLearnerImageSaver
    save_hipposlam_pth = join(run_dir, 'lib.pickle')
    save_trajdata_pth = join(run_dir, 'trajdata.pickle')
    save_img_dir = join(run_dir, 'imgs')
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

def preprocess_trajdata(run_dir):
    """
    Concatenate a list of episode data and generate a dataframe.
    Load:
        trajdata.pickle
    Save:
        trajdf.pickle
    """
    load_trajdata_pth = join(run_dir, 'trajdata.pickle')
    keys = ['x', 'y', 'a', 't', 'id_list', 'c']
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
    trajdf.to_pickle(join(run_dir, 'trajdf.pickle'))

def convert_images_to_mobilenet_embeddings(run_dir):
    """
    Load:
        trajdf.pickle
        imgs/*.png
    Save:
        embeds_index.pickle
        mobilenet_embeds.pt
        annotations.csv
    """
    # Paths
    load_trajdf_pth = join(run_dir, 'trajdf.pickle')
    load_img_dir = join(run_dir, 'imgs')

    # Load embeddings
    trajdf = read_pickle(load_trajdf_pth)
    img_mask = trajdf['img_exist']
    subdf = trajdf[img_mask]
    img_name_list = subdf['img_name'].tolist()

    # Save annotations and indexes
    annotations = subdf[['x', 'y', 'a']].reset_index(drop=True)
    annotations.to_csv(join(run_dir, 'annotations.csv'))
    embed_index = {subdf.index[i]: i for i in range(len(subdf.index))}
    save_pickle(join(run_dir, 'embeds_index.pickle'), embed_index)

    # Infer embeddings
    num_imgs = len(img_name_list)
    embeds = torch.zeros((num_imgs, 576))
    embedder = MobileNetEmbedder()
    for i, img_name in enumerate(tqdm(img_name_list)):
        load_img_pth = join(load_img_dir, img_name)
        embeds[i, :] = embedder.infer_embedding_from_path(load_img_pth).squeeze()  # -> (576, )

    # Save embeddings
    torch.save(embeds, join(run_dir, 'mobilenet_embeds.pt'))


def convert_embeddings_mobilenet_to_umap(run_dir):
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
    # Paths
    save_umap_dir = join(run_dir, 'umap_params')
    os.makedirs(save_umap_dir, exist_ok=True)
    load_embeds_pth = join(run_dir, 'mobilenet_embeds.pt')
    load_annotations_pth = join(run_dir, 'annotations.csv')

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
    save_parametric_umap_model(umap_model, save_umap_dir, umins, umaxs)
    torch.save(umap_embeds, join(save_umap_dir, 'umap_embeddings.pt'))

    # Load Annotation
    ann_df = pd.read_csv(load_annotations_pth)

    # Plot Umap embedding space and save
    fig, ax = plot_umap_embeddings_2d(umap_embeds, ann_df, nneigh, min_dist, metric)
    fig.savefig(join(save_umap_dir, 'umap_embeds_vis.png'), dpi=300)


def check_trained_umap_model(run_dir):
    """
    Load:
        mobilenet_embeds.pt
        annotations.csv
        save_umap_dir/* (except umap_embeddings.pt and *.png)
    Save:
        save_umap_dir/umap_embeds_vis_check.png

    """
    load_umap_dir = join(run_dir, 'umap_params')
    os.makedirs(load_umap_dir, exist_ok=True)
    load_embeds_pth = join(run_dir, 'mobilenet_embeds.pt')
    load_annotations_pth = join(run_dir, 'annotations.csv')


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
    im0 = ax[0].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['x'], cmap='jet')
    cbar0 = plt.colorbar(im0, ax=ax[0])
    cbar0.ax.set_label('x (m)')

    im1 = ax[1].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['y'], cmap='jet')
    cbar1 = plt.colorbar(im1, ax=ax[1])
    cbar1.ax.set_label('y (m)')

    im2 = ax[2].scatter(umap_embeds[:, 0], umap_embeds[:, 1], s=1, c=ann_df['a'], cmap='hsv')
    cbar2 = plt.colorbar(im2, ax=ax[2])
    cbar2.ax.set_label(r'$\theta$ (rad)')

    txt = f'nneigh={nneigh}\nmindist={min_dist:0.2f}\nmetric={metric}\n' if nneigh is not None else ''
    ax[3].annotate(text=txt, xy=(0.2, 0.5), xycoords='axes fraction')
    ax[3].axis('off')
    fig.tight_layout()
    return fig, ax

def statemap_learn(run_dir):
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

    # Paths
    load_trajdf_pth = join(run_dir, 'trajdf.pickle')
    load_embedsIndex_pth = join(run_dir, 'embeds_index.pickle')
    load_umap_dir = join(run_dir, 'umap_params')
    load_umapEmbeds_pth = join(load_umap_dir, 'umap_embeddings.pt')

    # Load data
    trajdf = read_pickle(load_trajdf_pth)
    embeds_index = read_pickle(load_embedsIndex_pth)
    umap_embeddings = torch.load(load_umapEmbeds_pth)
    umins = torch.load(join(load_umap_dir, 'umins.pt'))
    umaxs = torch.load(join(load_umap_dir, 'umaxs.pt'))


    # Initialize lib
    hipposeq = Sequences(R=R, L=L, reobserve=False)
    hippomap = StateDecoder(R=R, L=L, maxN=1000, area_norm=True)
    hippomap.set_lowSthresh(0.98)

    # Prepare for storage
    df_cols = ['pred_sid', 'targ_sid', 'img_name', 'img_exist', 'x', 'y', 'a']
    df_dict = {key:[] for key in df_cols}

    # Infer the state
    for i in tqdm(range(trajdf.shape[0])):
        id_list, x, y, a, img_name, img_exist = trajdf.loc[i, ['id_list', 'x', 'y', 'a', 'img_name', 'img_exist']]

        hipposeq.step(id_list)

        if (i in embeds_index) and (hippomap.current_F > 0):
            umap_embed = umap_embeddings[embeds_index[i], :]  # -> (2, )

            current_embedid = hippomap.learn_embedding(hipposeq.X, umap_embed, umins, umaxs,
                                                                 far_ids=None)
            df_dict['targ_sid'].append(current_embedid)
        else:
            df_dict['targ_sid'].append(-1)

        sid, Snodes = hippomap.infer_state(hipposeq.X)
        df_dict['pred_sid'].append(sid)
        df_dict['x'].append(x)
        df_dict['y'].append(y)
        df_dict['a'].append(a)
        df_dict['img_name'].append(img_name)
        df_dict['img_exist'].append(img_exist)

    df = pd.DataFrame(df_dict)
    df.to_csv(join(run_dir, 'simdf.csv'))


def analyze_state_specificity(run_dir):
    # Paths
    load_simdf_pth = join(run_dir, 'simdf.csv')
    save_dir = join(run_dir, 'state_specificity')
    os.makedirs(save_dir, exist_ok=True)

    # Load dataframe
    simdf = pd.read_csv(load_simdf_pth)

    simdf['offset'] = simdf['img_name'].apply(lambda x: int(x.split('.')[0].split('_')[1])) % 5
    embeddf = simdf[simdf['img_exist']]
    embedids = embeddf['targ_sid'].to_numpy()
    unique_embedids = np.unique(embedids)

    print(f'Num unique states = {unique_embedids.shape[0]}')
    aedges = np.linspace(-np.pi, np.pi, 16)

    xbound = (-18, 8)
    ybound = (-8, 18)

    for i, embedid_each in enumerate(tqdm(unique_embedids)):
        subdf = embeddf[embeddf['targ_sid'] == embedid_each]
        xya_targs = subdf[['x', 'y', 'a']].to_numpy()
        subsimdf = simdf[simdf['pred_sid'] == embedid_each]
        xya_pred = subsimdf[['x', 'y', 'a']].to_numpy()
        pred_offsets = subsimdf['offset'].to_numpy()
        title = f'embedid={embedid_each}, num={xya_pred.shape[0]}'
        fig, _ = compare_spatial_specificity(xya_targs, xya_pred, pred_offsets, aedges, xbound, ybound, title=title)

        fig.savefig(join(save_dir, '%d.png'%(i)), dpi=200)
        plt.close(fig)


