from dataclasses import dataclass

@dataclass
class TextDecoderConfig:
    vocab_size: int
    hidden_dim: int = 512
    num_decoder_layers: int = 1
    input_dim: int = 512
    max_seq_len: int = 20
