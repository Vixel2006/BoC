from dataclasses import dataclass

@dataclass
class BagOfConceptsConfig:
    num_concepts: int = 512
    concept_dim: int = 512
