from dataclasses import dataclass
from .image_encoder_config import ImageEncoderConfig
from .text_encoder_config import TextEncoderConfig
from .bag_of_concepts_config import BagOfConceptsConfig

@dataclass
class ConceptMapperConfig:
    modality: str
    image_encoder_config: ImageEncoderConfig = ImageEncoderConfig()
    text_encoder_config: TextEncoderConfig = TextEncoderConfig()
    bag_of_concepts_config: BagOfConceptsConfig = BagOfConceptsConfig()
    num_attention_heads: int = 4
    multi_head_attention_dropout_rate: float = 0.3
