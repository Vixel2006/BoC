from dataclasses import dataclass

@dataclass
class TextEncoderConfig:
    model_name: str = "distilroberta-base"
    target_hidden_size: int = 512
