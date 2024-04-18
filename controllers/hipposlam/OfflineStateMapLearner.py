import os

from lib.OfflinePipelines import preprocess_trajdata, convert_images_to_mobilenet_embeddings, \
    convert_embeddings_mobilenet_to_umap, check_trained_umap_model, ImageSampling, statemap_learn, \
    analyze_state_specificity
from os.path import join


def main(run_dir):
    # parameters
    assets_dir = join(run_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # 0: Simulation
    ImageSampling(assets_dir)


    # 1: Preprocess simulation data
    load_trajdata_pth = join(assets_dir, 'trajdata.pickle')
    preprocess_trajdata(assets_dir, load_trajdata_pth)

    # 2: MobileNet Embeddings
    load_img_dir = join(assets_dir, 'imgs')
    load_trajdf_pth = join(assets_dir, 'trajdf.pickle')
    convert_images_to_mobilenet_embeddings(assets_dir, load_trajdf_pth, load_img_dir)

    # 3 Umap Embeddings
    load_embeds_pth = join(assets_dir, 'mobilenet_embeds.pt')
    load_annotations_pth = join(assets_dir, 'annotations.csv')
    save_umap_dir = join(assets_dir, 'umap_params')
    os.makedirs(save_umap_dir, exist_ok=True)
    convert_embeddings_mobilenet_to_umap(load_embeds_pth, load_annotations_pth, save_umap_dir)

    # 4 Check if Umap can be loaded correctly
    load_embeds_pth = join(assets_dir, 'mobilenet_embeds.pt')
    load_annotations_pth = join(assets_dir, 'annotations.csv')
    load_umap_dir = join(assets_dir, 'umap_params')
    check_trained_umap_model(load_embeds_pth, load_annotations_pth, load_umap_dir)


    # Run Simulation
    load_trajdf_pth = join(assets_dir, 'trajdf.pickle')
    load_embedsIndex_pth = join(assets_dir, 'embeds_index.pickle')
    load_umap_dir = join(assets_dir, 'umap_params')
    load_umapEmbeds_pth = join(load_umap_dir, 'umap_embeddings.pt')
    statemap_learn(assets_dir, load_trajdf_pth, load_embedsIndex_pth, load_umapEmbeds_pth, load_umap_dir)


    # # Analyze simulation result
    load_simdf_pth = join(assets_dir, 'simdf.csv')
    analyze_state_specificity(assets_dir, load_simdf_pth)

if __name__ == '__main__':
    project_dir = r"D:\\data"
    experiment_name = 'OfflineStateMapLearner_IdList3'
    run_name = 'base'
    run_dir = join(project_dir, experiment_name, run_name)
    os.makedirs(run_dir, exist_ok=True)

    main(run_dir)
