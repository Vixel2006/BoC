import torch
import torch.nn as nn
from typing import Optional

from .encoders import ImageEncoder, TextEncoder
from .concepts_book import ConceptBook
from .concept_encoder import MultiHeadAttention

class MultimodalConceptMapper(nn.Module):
    def __init__(
        self,
        image_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "distilroberta-base",
        target_hidden_size: int = 512,
        num_concepts: int = 512,
        concept_dim: int = 512,
        num_attention_heads: int = 4
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(model_name=image_model_name, target_hidden_size=target_hidden_size)
        self.text_encoder = TextEncoder(model_name=text_model_name, target_hidden_size=target_hidden_size)
        self.concept_book = ConceptBook(num_concepts=num_concepts, concept_dim=concept_dim)

        self.multi_head_attention = MultiHeadAttention(
            query_dim=concept_dim,
            kv_dim=target_hidden_size,
            n_head=num_attention_heads
        )

    def forward(self, images: torch.Tensor, texts: list[str], concept_ids: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images) # (batch, seq_len_img, target_hidden_size)
        text_features = self.text_encoder(texts)   # (batch, seq_len_txt, target_hidden_size)

        # TODO:: I think a good idea here is to have a method to dropout certain modalities representation for some concepts
        # I think this can make the model more robust and make it less depedent on one modality
        kv_features = torch.cat((image_features, text_features), dim=1) # (batch, seq_len_img + seq_len_txt, target_hidden_size)

        concept_embedding = self.concept_book(concept_ids)
        query_input = concept_embedding.unsqueeze(1) # (batch, 1, concept_dim)

        attended_concept_representation = self.multi_head_attention(
            query_input=query_input,
            kv_input=kv_features,
        )
        # Squeeze the sequence dimension if the query was a single vector
        attended_concept_representation = attended_concept_representation.squeeze(1) # (batch, concept_dim)

        return attended_concept_representation
