import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, CLIPVisionModel
from ...configs.image_encoder_config import ImageEncoderConfig
from ...configs.text_encoder_config import TextEncoderConfig

class ImageEncoder(nn.Module):
    def __init__(self, config: ImageEncoderConfig = ImageEncoderConfig()):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(config.model_name)

        # Freezing the pre-trained image encoder
        for p in self.model.parameters(): p.requires_grad = False

        hidden_size = self.model.config.hidden_size

        if hidden_size != config.target_hidden_size:
            self.proj = nn.Linear(hidden_size, config.target_hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # NOTE: Here we just get the final hidden layer of the encoding,
        # we should study the possibility of pooling different layers results for more compact representations
        output = self.model(pixel_values=images, return_tensors="pt").last_hidden_state

        if hasattr(self, 'proj'):
            output = self.proj(output)

        return output


class TextEncoder(nn.Module):
    def __init__(self, config: TextEncoderConfig = TextEncoderConfig()):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = RobertaModel.from_pretrained(config.model_name)

        # Freezing the pre-trained text encoder
        for p in self.model.parameters(): p.requires_grad = False


        hidden_size = self.model.config.hidden_size

        if hidden_size != config.target_hidden_size:
            self.proj = nn.Linear(hidden_size, config.target_hidden_size)

    def forward(self, texts: torch.Tensor) -> torch.Tensor:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        # NOTE: Here we just get the final hidden layer of the encoding,
        # we should study the possibility of pooling different layers results for more compact representations
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        if hasattr(self, 'proj'):
            output = self.proj(output)

        return output
