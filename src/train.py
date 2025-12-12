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

from ..configs.training_config import TrainingConfig
from .models.concept_encoder import ConceptEncoder
from .models.bag_of_concepts import BagOfConcepts
from .models.decoders import SlotImageDecoder, SlotTextDecoderGRU
from .utils.visualization import plot_concept_alignment, plot_attention_map, plot_metric_per_epoch

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

@hydra.main(config_path="../configs", config_name="training_config", version_base=None)
def main_hydra(cfg: DictConfig):
    # Instantiate models
    txt_concept_encoder = instantiate(cfg.concept_mapper_text).to(cfg.device)
    img_concept_encoder = instantiate(cfg.concept_mapper_image).to(cfg.device)
    boc = instantiate(cfg.bag_of_concepts).to(cfg.device)
    img_decoder = instantiate(cfg.image_decoder).to(cfg.device)
    txt_decoder = instantiate(cfg.text_decoder).to(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    clip_mmd_calculator = CLIPMMD(device=cfg.device)

    # Instantiate optimizer (assuming it's defined in the config)
    optimizer = instantiate(cfg.optimizer, params=[
        *txt_concept_encoder.parameters(),
        *img_concept_encoder.parameters(),
        *boc.parameters(),
        *img_decoder.parameters(),
        *txt_decoder.parameters(),
    ])

    # Instantiate dataloader (assuming it's defined in the config)
    dataloader = instantiate(cfg.dataloader)

    # Create a dummy output directory for now
    import os
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    train(optimizer, dataloader, cfg, output_dir)



