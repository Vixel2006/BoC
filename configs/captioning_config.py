from dataclasses import dataclass
from typing import List

@dataclass
class CaptioningConfig:
    vocab_size: int
    concept_dim: int
    embed_size: int
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    self_attention_dropout_rate: float = 0.0
    src_attention_dropout_rate: float = 0.0
    use_output_layer: bool = True
    normalize_before: bool = True
    concat_after: bool = False
    ignore_id: int = 0