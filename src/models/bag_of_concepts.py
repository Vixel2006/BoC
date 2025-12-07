import torch
import torch.nn as nn
from ...configs.bag_of_concepts_config import BagOfConceptsConfig

class BagOfConcepts(nn.Module):
    def __init__(self, config: BagOfConceptsConfig = BagOfConceptsConfig()):
        super().__init__()
        # NOTE: Here I think we need to choose the num_concepts, concept carefully, like if we're training on FashionMnist with 10 labels,
        # maybe num_concepts=10 will yeld the best results as it will force the model to only have 10 concept vectors each one for a different labels
        # in which it can reconstruct the label or image from it. but this maybe will lead to overfitting, so we need to find a good balance of this
        self.concepts = nn.Embedding(config.num_concepts, config.concept_dim)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # NOTE: Here we should be aware that this concept bag can take any form of modality as input
        concept = self.concepts(inp)
        return concept

