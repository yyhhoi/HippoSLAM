from hipposlam.Embeddings import TrainContrastiveVAE, convert_to_embed
import logging
if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    TrainContrastiveVAE(1)
    # convert_to_embed(load_img_dir='data/VAE/imgs3',
    #                  load_annotation_pth='data/VAE/annotations3.csv',
    #                  save_embed_dir='data/VAE/embeds3', all=True)