import numpy as np

import torch
from torchvision import models

class WebotImageConvertor:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def to_torch_RGB(self, imagebytes):
        out = torch.tensor(bytearray(imagebytes)).reshape((self.height, self.width, 4)) / 255
        out = torch.flip(out, dims=[2])  # Convert the last channel to ARGB
        out = out[:, :, 1:]  # Use only RGB channels. Discard alpha channel
        out = torch.permute(out, [2, 0, 1])  # -> (RGB, height, width) as required
        return out


class MobileNetEmbedder:
    def __init__(self):
        self.model, self.preprocess = self._get_model()
    def infer_embedding(self, img_tensor):
        batch_t = self.preprocess(img_tensor).unsqueeze(0)

        # Get the features from the model
        with torch.no_grad():
            x = self.model.features(batch_t)
            x = self.model.avgpool(x)
            embedding = torch.flatten(x)
        return embedding.numpy()

    def _get_model(self):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        return model, preprocess
