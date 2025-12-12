import torch
import torch.nn as nn
from typing import Optional

from .encoders import ImageEncoder, TextEncoder
from .bag_of_concepts import BagOfConcepts
from .transformer_blocks import SlotAttention
from configs.concept_mapper_config import ConceptMapperConfig


class ConceptEncoder(nn.Module):
    def __init__(self, config: ConceptMapperConfig):
        super().__init__()
        self.modality = config.modality

        if self.modality == "text":
            self.encoder = TextEncoder(config.text_encoder_config)
        elif self.modality == "image":
            self.encoder = ImageEncoder(config.image_encoder_config)

        # NOTE: maybe if we found some problems in the text mapping we can add a standard attention
        # and use it for the text encoded tensors.
        self.slot_attn = SlotAttention(config.slot_dim, config.num_iter, config.num_slots)

    def forward(self, images: torch.Tensor = None, text: list[str] = None) -> torch.Tensor:
        # Make sure that the mapper gets the right input modality
        assert self.modality == "image" and (images is None or text is not None), "Image concept mapper can't be used for text"
        assert self.modality == "text" and (text is None or imagse is not None), "Text concept mapper can't be used for images"

        if self.modality == "image":
            features = self.encoder(images)
        if self.modality == "text":
            features = self.encoder(text)

        concept_slots, attn = self.slot_attn(features)

        return concept_slots, attn

