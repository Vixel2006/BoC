import torch
import torch.nn as nn
from ...configs.bag_of_concepts_config import BagOfConceptsConfig

class BagOfConcepts(nn.Module):
    def __init__(self, config: BagOfConceptsConfig = BagOfConceptsConfig()):
        super().__init__()
        self.config = config
        self.commitment = config.commitment
        self.concepts = nn.Embedding(config.num_concepts, config.concept_dim)
        self.concepts.weight.data.uniform_(-1/self.num_concepts, 1/self.num_concepts)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # NOTE: Here we should be aware that this concept bag can take any form of modality as input
        flat_concept = inp.reshape(-1, self.config.concept_dim)

        distances = torch.cdist(flat_concept, self.concepts.weight)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized_flat = torch.index_select(self.concepts.weight, 0, encoding_indices.view(-1))

        quantized = quantized_flat.view_as(inputs)

        return concept

