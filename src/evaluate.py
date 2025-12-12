import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import argparse
import os

from transformers import AutoTokenizer

from configs.training_config import TrainingConfig
from src.models.concept_encoder import ConceptEncoder
from src.models.bag_of_concepts import BagOfConcepts
from src.models.decoders import SlotImageDecoder, SlotTextDecoderGRU
from src.utils.visualization import plot_concept_alignment, plot_attention_map
import os
import matplotlib.pyplot as plt

def load_models(config: TrainingConfig, device: torch.device):
    """Instantiates and loads models based on the provided configuration."""
    txt_concept_encoder = ConceptEncoder(config.concept_mapper_text).to(device)
    img_concept_encoder = ConceptEncoder(config.concept_mapper_image).to(device)
    boc = BagOfConcepts(config.bag_of_concepts).to(device)
    img_decoder = SlotImageDecoder(config.image_decoder).to(device)
    txt_decoder = SlotTextDecoderGRU(config.text_decoder).to(device)

    # TODO: Implement loading of pre-trained weights if available
    # For now, models are instantiated with default/random weights.
    # Example: model.load_state_dict(torch.load("path/to/checkpoint.pth"))

    txt_concept_encoder.eval()
    img_concept_encoder.eval()
    boc.eval()
    img_decoder.eval()
    txt_decoder.eval()

    return txt_concept_encoder, img_concept_encoder, boc, img_decoder, txt_decoder

def text_to_image_evaluation(text_input: str, models, tokenizer, img_transform, inv_img_transform, device: torch.device, output_dir: str):
    txt_concept_encoder, _, boc, img_decoder, _, = models

    with torch.no_grad():
        # Encode text
        # The TextEncoder expects a list of strings, but ConceptEncoder's forward expects `text` as a list of strings
        # and internally tokenizes it.
        # So, we pass the raw text string in a list.
        text_features, txt_attn = txt_concept_encoder(text=[text_input])

        # Get concepts and indices
        txt_concepts, txt_indices = boc(text_features)

        # Decode image
        generated_image_tensor, _ = img_decoder(txt_concepts) # img_decoder returns (image, alpha)
        
        # Convert to PIL Image
        generated_image_tensor = generated_image_tensor.squeeze(0).cpu() # Remove batch dim, move to CPU
        generated_image_tensor = inv_img_transform(generated_image_tensor) # Inverse normalize
        generated_image_tensor = torch.clamp(generated_image_tensor, 0, 1) # Clamp to [0, 1]
        generated_image_pil = transforms.ToPILImage()(generated_image_tensor)
    
    print(f"Generated image for text: '{text_input}'")
    generated_image_pil.save(os.path.join(output_dir, "generated_image_t2i.png")) # Save generated image
    generated_image_pil.show() # Display the image

    # Plot attention map
    txt_attn_fig = plot_attention_map(txt_attn[0].squeeze(0), title=f"Text Attention Map (T2I)")
    txt_attn_fig.savefig(os.path.join(output_dir, f"txt_attn_t2i.png"))
    plt.close(txt_attn_fig)

    # Plot concept alignment
    alignment_fig = plot_concept_alignment(None, txt_indices) # img_indices is None for text-to-image
    alignment_fig.savefig(os.path.join(output_dir, f"concept_alignment_t2i.png"))
    plt.close(alignment_fig)

    return txt_indices, generated_image_pil

def image_to_text_evaluation(image_path: str, models, tokenizer, img_transform, device: torch.device, output_dir: str):
    _, img_concept_encoder, boc, _, txt_decoder = models

    # Load and transform image
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = img_transform(image_pil).unsqueeze(0).to(device) # Add batch dim

    with torch.no_grad():
        # Encode image
        image_features, img_attn = img_concept_encoder(images=image_tensor)

        # Get concepts and indices
        img_concepts, img_indices = boc(image_features)

        # Decode text
        generated_text_logits = txt_decoder(img_concepts)
        predicted_token_ids = torch.argmax(generated_text_logits, dim=-1).cpu()
        generated_text = tokenizer.decode(predicted_token_ids.squeeze(0), skip_special_tokens=True) # Remove batch dim

    print(f"Generated text for image '{image_path}': '{generated_text}'")

    # Plot attention map
    img_attn_fig = plot_attention_map(img_attn[0].squeeze(0), title=f"Image Attention Map (I2T)")
    img_attn_fig.savefig(os.path.join(output_dir, f"img_attn_i2t.png"))
    plt.close(img_attn_fig)

    # Plot concept alignment
    alignment_fig = plot_concept_alignment(img_indices, None) # txt_indices is None for image-to-text
    alignment_fig.savefig(os.path.join(output_dir, f"concept_alignment_i2t.png"))
    plt.close(alignment_fig)

    return img_indices, generated_text

@hydra.main(config_path="../configs", config_name="training_config", version_base=None)
def main_hydra(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Instantiate models
    txt_concept_encoder = instantiate(cfg.concept_mapper_text).to(device)
    img_concept_encoder = instantiate(cfg.concept_mapper_image).to(device)
    boc = instantiate(cfg.bag_of_concepts).to(device)
    img_decoder = instantiate(cfg.image_decoder).to(device)
    txt_decoder = instantiate(cfg.text_decoder).to(device)

    # TODO: Implement loading of pre-trained weights if available
    # For now, models are instantiated with default/random weights.
    # Example: model.load_state_dict(torch.load("path/to/checkpoint.pth"))

    txt_concept_encoder.eval()
    img_concept_encoder.eval()
    boc.eval()
    img_decoder.eval()
    txt_decoder.eval()

    models = (txt_concept_encoder, img_concept_encoder, boc, img_decoder, txt_decoder)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Image transformations for evaluation
    img_transform = transforms.Compose([
        transforms.Resize((cfg.image_decoder.img_size, cfg.image_decoder.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    inv_img_transform = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )

    # Create output directory for evaluation plots
    output_dir = "evaluation_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Example usage:
    # text_input = "a cat sitting on a couch"
    # txt_indices, generated_image_pil = text_to_image_evaluation(text_input, models, tokenizer, img_transform, inv_img_transform, device, output_dir)
    # plot_concept_alignment(None, txt_indices) # img_indices is None for text-to-image

    # image_path = "path/to/your/image.jpg"
    # img_indices, generated_text = image_to_text_evaluation(image_path, models, tokenizer, img_transform, device, output_dir)
    # plot_concept_alignment(img_indices, None) # txt_indices is None for image-to-text


