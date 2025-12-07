from dataclasses import dataclass

@dataclass
class ImageEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    target_hidden_size: int = 512
