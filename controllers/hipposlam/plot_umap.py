import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from os.path import join
import numpy as np
import torch
import umap
from hipposlam.trainVAE import get_dataloaders, VAELearner
from sklearn.decomposition import PCA

def plot_umap():
    # Paths
    model_tag = f'Annealing_bottle25'
    ckpt_name = 'ckpt_Annealing_bottle25_cycle9.pt'
    model_tag = 'OnlyEmbed'

    data_dir = join('data', 'VAE')
    load_embed_dir = join(data_dir, 'embeds2')
    load_annotation_pth = join(data_dir, 'annotations2.csv')
    save_dir = join(data_dir, 'model', model_tag)
    load_ckpt_pth = join(save_dir, ckpt_name)
    save_plot_dir = join(save_dir, 'umap_plots')
    os.makedirs(save_plot_dir, exist_ok=True)

    # Prepare datasets
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_dataloaders(load_annotation_pth, load_embed_dir)

    # Model
    dims = [400, 200, 100, 50, 25]
    vaelearner = VAELearner(input_dim=576,
                            hidden_dims=dims,
                            kld_mul=0.1,
                            lr=0.001,
                            lr_gamma=0.95,
                            weight_decay=0)
    # vaelearner.load_checkpoint(load_ckpt_pth)


    # Predict latents
    vaelearner.vae.eval()
    mu_train_tmp = []
    mu_test_tmp = []
    labels_train_tmp = []
    labels_test_tmp = []
    for x_train, label_train in iter(train_dataloader):
        _, _, (y, mu_train, _) = vaelearner.infer(x_train, kld_mul=1)
        # mu_train_tmp.append(mu_train)
        mu_train_tmp.append(x_train)
        labels_train_tmp.append(label_train)
    mus_train = torch.vstack(mu_train_tmp)
    labels_train = torch.vstack(labels_train_tmp)

    for x_test, label_test in iter(test_dataloader):
        _, _, (y, mu_test, _) = vaelearner.infer(x_test, kld_mul=1)
        # mu_test_tmp.append(mu_test)
        mu_test_tmp.append(x_test)
        labels_test_tmp.append(label_test)
    mus_test = torch.vstack(mu_test_tmp)
    labels_test = torch.vstack(labels_test_tmp)
    mus = torch.vstack([mus_train, mus_test])
    labels = torch.vstack([labels_train, labels_test])

    # Fit and plot UMAP

    nneighs = [10, 50, 100]
    min_dists = [0.1, 0.5, 0.9]
    metrics = ['cosine', 'euclidean']

    for nneigh in nneighs:
        for min_dist in min_dists:
            for metric in metrics:
                umap_tag = f'nneigh={nneigh}_mindist={min_dist:0.2f}_metric={metric}'
                print(umap_tag)


                umap_model = umap.UMAP(n_neighbors=nneigh,
                                       min_dist=min_dist,
                                       metric=metric,
                                       n_components=2)
                umap_embeds = umap_model.fit_transform(mus)
                umap_embeds_toplot = umap_embeds
                labels_toplot = labels
                fig = plt.figure(figsize=(12, 8))
                ax0 = fig.add_subplot(2, 2, 1)
                ax1 = fig.add_subplot(2, 2, 2)
                ax2 = fig.add_subplot(2, 2, 3)
                ax3 = fig.add_subplot(2, 2, 4)
                im0 = ax0.scatter(umap_embeds_toplot[:, 0], umap_embeds_toplot[:, 1], s=1, c=labels_toplot[:, 0], cmap='jet', alpha=0.5)
                plt.colorbar(im0, ax=ax0)
                im1 = ax1.scatter(umap_embeds_toplot[:, 0], umap_embeds_toplot[:, 1], s=1, c=labels_toplot[:, 1], cmap='jet', alpha=0.5)
                plt.colorbar(im1, ax=ax1)
                im2 = ax2.scatter(umap_embeds_toplot[:, 0], umap_embeds_toplot[:, 1], s=1, c=labels_toplot[:, 2], cmap='hsv', alpha=0.5)
                plt.colorbar(im2, ax=ax2)

                txt = f'nneigh={nneigh}\nmindist={min_dist:0.2f}\nmetric={metric}\nkldmul={vaelearner.kld_mul:0.6f}\n' + \
                      f'dims={str(dims)}\n'
                ax3.annotate(text=txt, xy =(0.2, 0.5), xycoords='axes fraction')
                ax3.axis('off')
                fig.tight_layout()
                fig.savefig(join(save_plot_dir, f'Umap_{umap_tag}.png'), dpi=300)
                plt.close(fig)

    pca = PCA()
    pca.fit(mus)
    varr = pca.explained_variance_ratio_
    fig, ax = plt.subplots()
    ax.scatter(np.arange(varr.shape[0]), varr)
    fig.savefig(join(save_plot_dir, 'PCA.png'), dpi=300)


if __name__ == '__main__':
    # for kld_mul in [0.1, 0.01, 0.001]:
    #     print(f'\n {kld_mul:0.6f}')
    #     plot_umap(kld_mul)
    #     print()

    plot_umap()