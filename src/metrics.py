import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np

# Helper function for MMD
def gaussian_kernel(x, y, gamma=1.0):
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-gamma * dist_sq)

class CLIPScore(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode

    def forward(self, images, texts):
        if not (isinstance(images, list) and all(isinstance(img, Image.Image) for img in images)):
            raise ValueError("Images must be a list of PIL.Image.Image objects.")
        if not isinstance(texts, list):
            raise ValueError("Texts must be a list of strings.")

        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Calculate cosine similarity
        # Assuming images[i] corresponds to texts[i]
        clip_scores = (image_features * text_features).sum(dim=-1)
        return clip_scores


class CLIPMMD(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", kernel_gamma=1.0, device=None):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.kernel_gamma = kernel_gamma

    def forward(self, images_a, texts_b):
        if not (isinstance(images_a, list) and all(isinstance(img, Image.Image) for img in images_a)):
            raise ValueError("images_a must be a list of PIL.Image.Image objects.")
        if not isinstance(texts_b, list):
            raise ValueError("texts_b must be a list of strings.")

        # Get image embeddings
        inputs_a = self.processor(images=images_a, return_tensors="pt")
        inputs_a = {k: v.to(self.device) for k, v in inputs_a.items()}
        with torch.no_grad():
            image_features_a = self.model.get_image_features(**inputs_a)
            image_features_a = F.normalize(image_features_a, p=2, dim=-1)

        # Get text embeddings
        inputs_b = self.processor(text=texts_b, return_tensors="pt", padding=True)
        inputs_b = {k: v.to(self.device) for k, v in inputs_b.items()}
        with torch.no_grad():
            text_features_b = self.model.get_text_features(**inputs_b)
            text_features_b = F.normalize(text_features_b, p=2, dim=-1)

        # Calculate MMD
        mmd_score = self._mmd_rbf(image_features_a, text_features_b)
        return mmd_score

    def _mmd_rbf(self, x, y):
        xx = gaussian_kernel(x, x, gamma=self.kernel_gamma).mean()
        yy = gaussian_kernel(y, y, gamma=self.kernel_gamma).mean()
        xy = gaussian_kernel(x, y, gamma=self.kernel_gamma).mean()
        return xx + yy - 2 * xy
