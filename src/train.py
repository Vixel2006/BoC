import torch
import torch.nn.functional as F
from .losses import *
from .metrics import CLIPMMD
from PIL import Image
from torchvision import transforms
import numpy as np
from bert_score import score
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import torch.optim as optim

from configs.training_config import TrainingConfig
from configs.bag_of_concepts_config import BagOfConceptsConfig
from configs.concept_mapper_config import ConceptMapperConfig
from configs.image_decoder_config import ImageDecoderConfig
from configs.text_decoder_config import TextDecoderConfig

from .models.concept_encoder import ConceptEncoder
from .models.bag_of_concepts import BagOfConcepts
from .models.decoders import SlotImageDecoder, SlotTextDecoderGRU
from .utils.visualization import plot_concept_alignment, plot_attention_map, plot_metric_per_epoch
from data.loader import flickr30k_loader

def train(optimizer, dataloader, config: TrainingConfig, output_dir: str):
    # Instantiate models
    txt_concept_encoder = ConceptEncoder(config.concept_mapper_text).to(config.device)
    img_concept_encoder = ConceptEncoder(config.concept_mapper_image).to(config.device)
    boc = BagOfConcepts(config.bag_of_concepts).to(config.device)
    img_decoder = SlotImageDecoder(config.image_decoder).to(config.device)
    txt_decoder = SlotTextDecoderGRU(config.text_decoder).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    clip_mmd_calculator = CLIPMMD(device=config.device)

    # Lists to store metrics for plotting
    total_loss_per_epoch = []
    alignment_loss_per_epoch = []
    txt_recon_loss_per_epoch = []
    img_recon_loss_per_epoch = []
    bertscore_f1_per_epoch = []
    clip_mmd_per_epoch = []

    for epoch in range(config.epochs):
        # Wrap the dataloader with tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for imgs, txts in pbar:
            imgs, txts = imgs.to(config.device), txts.to(config.device)

            optimizer.zero_grad()

            # NOTE: The training loop is: Modality specific encoder -> Concept encoder -> Modality specific decoder
            img_features, img_attn = img_concept_encoder(imgs)
            txt_features, txt_attn = txt_concept_encoder(txts)

            txt_concept, txt_indices = boc(txt_features)
            img_concept, img_indices = boc(img_features)

            gen_img = img_decoder(img_concept)
            gen_txt = txt_decoder(txt_concept, txts)

            # NOTE: The loss function used consists of three parts
            # 1. We have alignment_loss, that align each encoder output to a concept and try to align both modalities concepts
            # 2. We have a reconstruction loss for each modality to make sure that we can get the modality back out of its corresponding concept
            # The 'slots' variable is the codebook/centroids for vector quantization.
            # It should ideally be part of the BagOfConcepts model's state.
            # For now, we'll use a random tensor as a placeholder if boc.slots is not available.
            slots = boc.slots if hasattr(boc, 'slots') else torch.randn(config.bag_of_concepts.num_concepts, config.bag_of_concepts.concept_dim).to(config.device)
            alignment_loss = vq_loss(txt_concept, slots, config.bag_of_concepts.commitment) + vq_loss(img_concept, slots, config.bag_of_concepts.commitment) + F.mse_loss(img_concept, txt_concept)

            txt_recon_loss = text_reconstruction_loss(gen_txt, txts)
            img_recon_loss = image_reconstruction_loss(imgs, gen_img)

            # TODO: Maybe we should add weights to each loss for better results
            loss = alignment_loss + txt_recon_loss + img_recon_loss

            loss.backward()

            optimizer.step()

            pbar.set_postfix(
                total_loss=loss.item(),
                alignment_loss=alignment_loss.item(),
                txt_recon_loss=txt_recon_loss.item(),
                img_recon_loss=img_recon_loss.item()
            )

            if epoch % 10 == 0:
                # --- Plotting ---
                # Plot attention maps
                img_attn_fig = plot_attention_map(img_attn[0].squeeze(0), title=f"Image Attention Map (Epoch {epoch})")
                img_attn_fig.savefig(os.path.join(output_dir, f"img_attn_epoch_{epoch}.png"))
                plt.close(img_attn_fig)

                txt_attn_fig = plot_attention_map(txt_attn[0].squeeze(0), title=f"Text Attention Map (Epoch {epoch})")
                txt_attn_fig.savefig(os.path.join(output_dir, f"txt_attn_epoch_{epoch}.png"))
                plt.close(txt_attn_fig)

                # Plot concept alignment
                alignment_fig = plot_concept_alignment(img_indices, txt_indices)
                alignment_fig.savefig(os.path.join(output_dir, f"concept_alignment_epoch_{epoch}.png"))
                plt.close(alignment_fig)
                # --- End Plotting ---

                # --- Metric Calculation ---
                # Convert image tensor to PIL Images
                # Assuming images were normalized with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                inv_normalize = transforms.Normalize(
                    mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
                    std=[1/0.5, 1/0.5, 1/0.5]
                )
                pil_images = []
                for img_tensor in imgs.cpu():
                    # Ensure 3 channels for RGB, even if input was grayscale
                    if img_tensor.shape[0] == 1:
                        img_tensor = img_tensor.repeat(3, 1, 1)
                    img_tensor = inv_normalize(img_tensor)
                    img_tensor = torch.clamp(img_tensor, 0, 1) # Clamp to [0, 1] before converting to uint8
                    img_tensor = img_tensor.permute(1, 2, 0) # C, H, W -> H, W, C
                    img_np = img_tensor.numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(img_np))

                # Decode text tensors to strings for BERTScore
                # NOTE: This assumes 'txts' and 'gen_txt' are tensors of token IDs
                # that can be decoded by the BERT tokenizer.
                reference_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in txts.cpu()]
                # 'gen_txt' is likely logits, so we need to get the predicted token IDs first
                predicted_token_ids = torch.argmax(gen_txt, dim=-1).cpu()
                candidate_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in predicted_token_ids]

                # Calculate BERTScore
                # Ensure candidate_texts and reference_texts are lists of strings
                P, R, F1 = score(candidate_texts, reference_texts, lang="en", verbose=False)
                print(f"Epoch {epoch}, BERTScore: P={P.mean().item():.4f}, R={R.mean().item():.4f}, F1={F1.mean().item():.4f}")
                bertscore_f1_per_epoch.append(F1.mean().item())

                # Calculate CLIP MMD
                # For MMD, we need two sets of data. Here, we'll use the same images and texts for simplicity
                # In a real scenario, you might compare generated images to real images, or generated text to real text.
                # For demonstration, we'll compare image features from pil_images to text features from reference_texts.
                # NOTE: CLIPMMD expects a list of PIL Images and a list of strings.
                clip_mmd_score = clip_mmd_calculator(pil_images, reference_texts)
                print(f"Epoch {epoch}, CLIP MMD Score: {clip_mmd_score.item()}")
                clip_mmd_per_epoch.append(clip_mmd_score.item())
                # --- End Metric Calculation ---
            
            # Store metrics per epoch
            total_loss_per_epoch.append(loss.item())
            alignment_loss_per_epoch.append(alignment_loss.item())
            txt_recon_loss_per_epoch.append(txt_recon_loss.item())
            img_recon_loss_per_epoch.append(img_recon_loss.item())

    # Plot metrics after training
    total_loss_fig = plot_metric_per_epoch(total_loss_per_epoch, "Total Loss", "Total Loss per Epoch")
    total_loss_fig.savefig(os.path.join(output_dir, "total_loss_per_epoch.png"))
    plt.close(total_loss_fig)

    alignment_loss_fig = plot_metric_per_epoch(alignment_loss_per_epoch, "Alignment Loss", "Alignment Loss per Epoch")
    alignment_loss_fig.savefig(os.path.join(output_dir, "alignment_loss_per_epoch.png"))
    plt.close(alignment_loss_fig)

    txt_recon_loss_fig = plot_metric_per_epoch(txt_recon_loss_per_epoch, "Text Reconstruction Loss", "Text Reconstruction Loss per Epoch")
    txt_recon_loss_fig.savefig(os.path.join(output_dir, "txt_recon_loss_per_epoch.png"))
    plt.close(txt_recon_loss_fig)

    img_recon_loss_fig = plot_metric_per_epoch(img_recon_loss_per_epoch, "Image Reconstruction Loss", "Image Reconstruction Loss per Epoch")
    img_recon_loss_fig.savefig(os.path.join(output_dir, "img_recon_loss_per_epoch.png"))
    plt.close(img_recon_loss_fig)

    if bertscore_f1_per_epoch: # Only plot if data exists
        bertscore_f1_fig = plot_metric_per_epoch(bertscore_f1_per_epoch, "BERTScore F1", "BERTScore F1 per Epoch")
        bertscore_f1_fig.savefig(os.path.join(output_dir, "bertscore_f1_per_epoch.png"))
        plt.close(bertscore_f1_fig)

    if clip_mmd_per_epoch: # Only plot if data exists
        clip_mmd_fig = plot_metric_per_epoch(clip_mmd_per_epoch, "CLIP MMD Score", "CLIP MMD Score per Epoch")
        clip_mmd_fig.savefig(os.path.join(output_dir, "clip_mmd_per_epoch.png"))
        plt.close(clip_mmd_fig)

    # Return collected metrics
    return {
        "total_loss_per_epoch": total_loss_per_epoch,
        "alignment_loss_per_epoch": alignment_loss_per_epoch,
        "txt_recon_loss_per_epoch": txt_recon_loss_per_epoch,
        "img_recon_loss_per_epoch": img_recon_loss_per_epoch,
        "bertscore_f1_per_epoch": bertscore_f1_per_epoch,
        "clip_mmd_per_epoch": clip_mmd_per_epoch,
    }

def main(config: TrainingConfig):
    # Instantiate models
    txt_concept_encoder = ConceptEncoder(config.concept_mapper_text).to(config.device)
    img_concept_encoder = ConceptEncoder(config.concept_mapper_image).to(config.device)
    boc = BagOfConcepts(config.bag_of_concepts).to(config.device)
    img_decoder = SlotImageDecoder(config.image_decoder).to(config.device)
    txt_decoder = SlotTextDecoderGRU(config.text_decoder).to(config.device)

    # Instantiate optimizer
    # Assuming Adam optimizer with a default learning rate if not specified in config
    optimizer = optim.Adam(
        [
            *txt_concept_encoder.parameters(),
            *img_concept_encoder.parameters(),
            *boc.parameters(),
            *img_decoder.parameters(),
            *txt_decoder.parameters(),
        ],
        lr=1e-3 # Default learning rate
    )

    # Instantiate dataloader
    # Assuming default transforms for now, as they are not in TrainingConfig
    transform = transforms.Compose([
        transforms.Resize((config.image_decoder.image_size, config.image_decoder.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataloader = flickr30k_loader(
        root="./data", # Assuming data is in a 'data' folder relative to project root
        ann_file="./data/flickr30k/annotations", # Adjust path as needed
        transforms=transform,
        batch_size=32, # Assuming a default batch size for now
        shuffle=True,
        num_workers=0, # For simplicity, can be increased
        pin_memory=True if config.device == "cuda" else False,
    )

    # Create output directory
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    # Run the training loop
    metrics = train(optimizer, dataloader, config, output_dir)
    return metrics

