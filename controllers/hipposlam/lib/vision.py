import numpy as np

import torch
from torchvision import models

import cv2 as cv
from torchvision.io import read_image


class WebotImageConvertor:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def to_torch_RGB(self, imagebytes):
        """
        Convert the image bytes from Webots.Camera.getImage() to torch.Tensor.

        Parameters
        ----------
        imagebytes : bytes
            Flattened bytes for an image. Can be converted to shape (height, width, BRGA channels) with dtype=uint8

        Returns
        -------
        Image array : torch.Tensor
            Shape = (RGB, height, width). dtype=torch.float32. Value range = [0.0, 1.0]

        """
        out = torch.tensor(bytearray(imagebytes)).reshape((self.height, self.width, 4))  # BGRA, torch.uint8
        out = out / 255  # dtype = torch.float32
        out = torch.flip(out, dims=[2])  # Convert the last channel to ARGB
        out = out[:, :, 1:]  # Use only RGB channels. Discard alpha channel
        out = torch.permute(out, [2, 0, 1])  # -> (RGB, height, width) as required
        return out


class MobileNetEmbedder:
    def __init__(self):
        self.model, self.preprocess = self._get_model()
    def infer_embedding(self, img_tensor):
        """
        Infer embedding with pretrained MobileNetV3 from a torch tensor (batched or unbatched).

        Parameters
        ----------
        img_tensor : torch.Tensor
            (3, 256, 256) or (N, 3, 256, 256). RGB channels. Either float or integers.
            dtype=torch.float32. Value range=[0.0, 1.0]. OR dtype=torch.uint8. Value range=[0, 1, ..., 255].

        Returns
        -------
        embedding : torch.Tensor
            (1, 576) or (N, 576). dtype=torch.float32.
        """

        if len(img_tensor.shape) == 3:
            batch_t = self.preprocess(img_tensor).unsqueeze(0)
        else:
            batch_t = img_tensor

        # Get the features from the model
        with torch.no_grad():
            x = self.model.features(batch_t)
            x = self.model.avgpool(x)
            embedding = torch.flatten(x)
        return embedding

    def infer_embedding_from_path(self, load_img_pth):
        img_tensor = read_image(load_img_pth)[:3, ...]
        embedding = self.infer_embedding(img_tensor)
        return embedding


    def _get_model(self):
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.eval()
        # See https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html#torchvision.models.mobilenet_v3_small
        # for details of the preprocessing pipelines
        preprocess = weights.transforms()
        return model, preprocess
