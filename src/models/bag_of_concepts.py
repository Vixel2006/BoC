import torch
import torch.nn as nn
from configs.bag_of_concepts_config import BagOfConceptsConfig

class BagOfConcepts(nn.Module):
    def __init__(self, config: BagOfConceptsConfig = BagOfConceptsConfig()):
        super().__init__()
        self.config = config
        self.commitment = config.commitment
        self.concepts = nn.Embedding(config.num_concepts, config.concept_dim)
        self.concepts.weight.data.uniform_(-1/self.num_concepts, 1/self.num_concepts)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        flat_inp = inp.reshape(-1, self.config.concept_dim) # (batch_size * num_slots, concept_dim)

        distances = torch.cdist(flat_inp, self.concepts.weight)

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        concepts_flat = torch.index_select(self.concepts.weight, 0, encoding_indices.view(-1))

        concepts = concepts_flat.view_as(inputs)

        # NOTE: Here we do this trick so we can pass the discrete concepts but,
        # make the inputs passed through the backward prop to avoid the non-differntiable nature of the argmin
        # here the detach makes it explicit that (concepts - inputs) is freezed in terms of the backward prop
        # This is equavelent to saying that dL/d(concepts) = dL/d(inputs)
        concepts = inputs + (concepts - inputs).detach()

        return concepts, encoding_indices

