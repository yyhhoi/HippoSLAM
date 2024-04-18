import os

from lib.OfflinePipelines import preprocess_trajdata, convert_images_to_mobilenet_embeddings, \
    convert_embeddings_mobilenet_to_umap, check_trained_umap_model, ImageSampling, statemap_learn, \
    analyze_state_specificity
from os.path import join


def main(run_dir):
    # Paths
    os.makedirs(run_dir, exist_ok=True)

    # # 1: Simulation
    # ====================== Uncomment below ===========================
    # If you use external IDE, uncomment the line below to sample images.
    # ImageSampling(run_dir)
    # ==================================================================
    # ====================== Uncomment below ===========================
    # If you use Webots bulti-in editor, please run ../image_sampling/image_sampling.py in Webots directly.
    # ImageSampling(run_dir)
    # ==================================================================

    # # 2: Preprocess simulation data
    # preprocess_trajdata(run_dir)
    #
    # # 3: MobileNet Embeddings
    # convert_images_to_mobilenet_embeddings(run_dir)
    #
    # # 4: Umap Embeddings
    # convert_embeddings_mobilenet_to_umap(run_dir)

    # # 5: Check if Umap can be loaded correctly, and if the loaded Umap produces the same result
    # check_trained_umap_model(run_dir)

    # # 6: Run Offline simulation and state inference with hipposlam
    # statemap_learn(run_dir)

    # # 7: Analyze simulation result
    # analyze_state_specificity(run_dir)

if __name__ == '__main__':
    experiment_dir = join('data', 'OfflineAnalysis')
    run_name = 'OfflineStateMapLearner'
    run_dir = join(experiment_dir, run_name)
    main(run_dir)
