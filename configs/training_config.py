from dataclasses import dataclass, field
import torch

from .bag_of_concepts_config import BagOfConceptsConfig
from .concept_mapper_config import ConceptMapperConfig
from .image_decoder_config import ImageDecoderConfig
from .text_decoder_config import TextDecoderConfig

@dataclass
class TrainingConfig:
    epochs: int = 100
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    bag_of_concepts: BagOfConceptsConfig = field(default_factory=BagOfConceptsConfig)

    concept_mapper_image: ConceptMapperConfig = field(default_factory=lambda: ConceptMapperConfig(modality="image"))
    concept_mapper_text: ConceptMapperConfig = field(default_factory=lambda: ConceptMapperConfig(modality="text"))

    image_decoder: ImageDecoderConfig = field(default_factory=ImageDecoderConfig)
    text_decoder: TextDecoderConfig = field(default_factory=lambda: TextDecoderConfig(vocab_size=0))
