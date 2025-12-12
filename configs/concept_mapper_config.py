from dataclasses import dataclass, field
from .image_encoder_config import ImageEncoderConfig
from .text_encoder_config import TextEncoderConfig
from .bag_of_concepts_config import BagOfConceptsConfig

@dataclass
class ConceptMapperConfig:
    modality: str
    image_encoder_config: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    text_encoder_config: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    bag_of_concepts_config: BagOfConceptsConfig = field(default_factory=BagOfConceptsConfig)
    num_attention_heads: int = 4
    multi_head_attention_dropout_rate: float = 0.3
