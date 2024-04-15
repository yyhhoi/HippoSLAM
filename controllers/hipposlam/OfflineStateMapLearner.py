import os

from hipposlam.offline_pipelines import preprocess_trajdata, convert_images_to_mobilenet_embeddings, \
    convert_embeddings_mobilenet_to_umap, check_trained_umap_model
from os.path import join


def main(run_dir):
    # parameters
    assets_dir = join(run_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # # 1: Preprocess simulation data
    # load_trajdata_pth = join(assets_dir, 'trajdata.pickle')
    # save_trajdf_pth = join(assets_dir, 'trajdf.pickle')
    # preprocess_trajdata(load_trajdata_pth, save_trajdf_pth)
    #
    # # 2: MobileNet Embeddings
    # load_img_dir = join(assets_dir, 'imgs')
    # load_trajdf_pth = join(assets_dir, 'trajdf.pickle')
    # save_annotation_pth = join(assets_dir, 'annotations.csv')
    # save_embeds_pth = join(assets_dir, 'mobilenet_embeds.pt')
    # convert_images_to_mobilenet_embeddings(load_trajdf_pth, load_img_dir, save_embeds_pth, save_annotation_pth)
    #
    # # 3 Umap Embeddings
    # load_embeds_pth = join(assets_dir, 'mobilenet_embeds.pt')
    # load_annotations_pth = join(assets_dir, 'annotations.csv')
    # save_umap_dir = join(assets_dir, 'umap_params')
    # os.makedirs(save_umap_dir, exist_ok=True)
    # convert_embeddings_mobilenet_to_umap(load_embeds_pth, load_annotations_pth, save_umap_dir)

    # # 4 Check if Umap can be loaded correctly
    # load_embeds_pth = join(assets_dir, 'mobilenet_embeds.pt')
    # load_annotations_pth = join(assets_dir, 'annotations.csv')
    # load_umap_dir = join(assets_dir, 'umap_params')
    # check_trained_umap_model(load_embeds_pth, load_annotations_pth, load_umap_dir)


if __name__ == '__main__':
    project_dir = r"D:\\data"
    experiment_name = 'OfflineStateMapLearner'
    run_name = 'base'
    run_dir = join(project_dir, experiment_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    main(run_dir)
