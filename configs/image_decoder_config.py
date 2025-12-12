from dataclasses import dataclass

@dataclass
class ImageDecoderConfig:
    image_size: int = 64
    num_decoder_layers: int = 4
    input_dim: int = 512
    hidden_dim: int = 512
