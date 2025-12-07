from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConceptToImageVAEConfig:
    concept_dim: int = 512
    output_channels: int = 3
    hidden_dims: Optional[List[int]] = None
    image_size: int = 64
