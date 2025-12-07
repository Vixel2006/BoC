import torch
import torch.nn as nn
from typing import Optional

from .encoders import ImageEncoder, TextEncoder
from .bag_of_concepts import BagOfConcepts
from .transformer_blocks import MultiHeadAttention
from ...configs.concept_mapper_config import ConceptMapperConfig

class MultimodalConceptMapper(nn.Module):
    def __init__(self, config: ConceptMapperConfig):
        super().__init__()

        self.image_encoder = ImageEncoder(config=config.image_encoder_config)
        self.text_encoder = TextEncoder(config=config.text_encoder_config)
        self.bag_of_concepts = BagOfConcepts(config=config.bag_of_concepts_config)

        self.multi_head_attention = MultiHeadAttention(
            query_dim=config.bag_of_concepts_config.concept_dim,
            kv_dim=config.image_encoder_config.target_hidden_size,
            n_head=config.num_attention_heads,
            dropout_rate=config.multi_head_attention_dropout_rate
        )

    def forward(self, images: torch.Tensor, texts: list[str], concept_ids: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images) # (batch, seq_len_img, target_hidden_size)
        text_features = self.text_encoder(texts)   # (batch, seq_len_txt, target_hidden_size)

        # TODO:: I think a good idea here is to have a method to dropout certain modalities representation for some concepts
        # I think this can make the model more robust and make it less depedent on one modality
        kv_features = torch.cat((image_features, text_features), dim=1) # (batch, seq_len_img + seq_len_txt, target_hidden_size)

        concept_embedding = self.bag_of_concepts(concept_ids)
        query_input = concept_embedding.unsqueeze(1) # (batch, 1, concept_dim)

        attended_concept_representation = self.multi_head_attention(
            query_input=query_input,
            kv_input=kv_features,
        )
        # Squeeze the sequence dimension if the query was a single vector
        attended_concept_representation = attended_concept_representation.squeeze(1) # (batch, concept_dim)

        return attended_concept_representation
