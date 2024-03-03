import numpy as np

import torch
from torchvision import models, transforms

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
class ImageEmbedder:
    def __init__(self):
        self.model, self.preprocess = self._get_model()

    def infer_embedding(self, img_tensor):
        img_t = self.preprocess(img_tensor)
        batch_t = torch.unsqueeze(img_t, 0)  # Add a batch dimension

        # Get the features from the model
        with torch.no_grad():
            features = self.model.features(batch_t)  # (1, 3, height, width) -> (1, 768, 7, 7)
            pooled_features = self.model.avgpool(features)  # (1, 768, 7, 7) -> (1, 768, 1, 1)
            embedding = torch.flatten(pooled_features)  # (1, 768, 1, 1) -> (768)
        return embedding

    def _get_model(self):
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        return model, preprocess


class MobileNetEmbedder(ImageEmbedder):
    def infer_embedding(self, img_tensor):
        batch_t = self.preprocess(img_tensor).unsqueeze(0)

        # Get the features from the model
        with torch.no_grad():
            x = self.model.features(batch_t)
            x = self.model.avgpool(x)
            embedding = torch.flatten(x)
        return embedding

    def _get_model(self):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        preprocess = weights.transforms()
        return model, preprocess

class QuantizedMobileNetEmbedder(ImageEmbedder):
    def infer_embedding(self, img_tensor):
        batch_t = self.preprocess(img_tensor).unsqueeze(0)

        # Get the features from the model
        with torch.no_grad():
            x = self.model.quant(batch_t)
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = self.model.dequant(x)
            embedding = torch.flatten(x)
        return embedding

    def _get_model(self):
        weights = models.quantization.MobileNet_V3_Large_QuantizedWeights.IMAGENET1K_QNNPACK_V1
        model = models.quantization.mobilenet_v3_large(weights=weights, quantize=True)
        model.eval()
        preprocess = weights.transforms()
        return model, preprocess