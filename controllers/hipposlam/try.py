from hipposlam.trainVAE import TrainContrastiveVAE
import logging
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    TrainContrastiveVAE(0.1)