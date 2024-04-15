import torch
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from umap.parametric_umap import ParametricUMAP
from tqdm import tqdm
from .Embeddings import save_parametric_umap_model, load_parametric_umap_model
from .utils import read_pickle
from .vision import MobileNetEmbedder



def preprocess_trajdata(load_trajdata_pth: str, save_trajdf_pth: str):
    """
    Concatenate a list of episode data and generate a dataframe.

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
    trajdf.to_pickle(save_trajdf_pth)
    return trajdf

def convert_images_to_mobilenet_embeddings(load_trajdf_pth, load_img_dir, save_embeds_pth, save_annotation_pth):
    # Load embeddings
    trajdf = read_pickle(load_trajdf_pth)
    img_mask = trajdf['img_exist']
    subdf = trajdf[img_mask]
    img_name_list = subdf['img_name'].tolist()

    # Save annotations
    annotations = subdf[['x', 'y', 'a']].reset_index(drop=True)
    annotations.to_csv(save_annotation_pth)

    # Infer embeddings
    num_imgs = len(img_name_list)
    embeds = torch.zeros((num_imgs, 576))
    embedder = MobileNetEmbedder()
    for i, img_name in enumerate(tqdm(img_name_list)):
        load_img_pth = join(load_img_dir, img_name)
        embeds[i, :] = embedder.infer_embedding_from_path(load_img_pth).squeeze()  # -> (576, )

    # Save embeddings
    torch.save(embeds, save_embeds_pth)
    return embeds


def convert_embeddings_mobilenet_to_umap(load_embeds_pth, load_annotations_pth, save_umap_dir):
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

    # Load Annotation
    ann_df = pd.read_csv(load_annotations_pth)

    # Plot Umap embedding space and save
    fig, ax = plot_umap_embeddings_2d(umap_embeds, ann_df, nneigh, min_dist, metric)
    fig.savefig(join(save_umap_dir, 'umap_embeds_vis.png'), dpi=300)


def check_trained_umap_model(load_embeds_pth, load_annotations_pth, load_umap_dir):

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