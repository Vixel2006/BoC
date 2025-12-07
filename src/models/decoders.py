import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .transformer_blocks import *
from ...configs.captioning_config import CaptioningConfig
from ...configs.concept_to_image_vae_config import ConceptToImageVAEConfig

class ConceptConditionedImageCaptioning(nn.Module):
    def __init__(self, config: CaptioningConfig):
        super().__init__()

        self.transformer_decoder = TransformerDecoder(
            vocab_size=config.vocab_size,
            encoder_output_size=config.concept_dim,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            self_attention_dropout_rate=config.self_attention_dropout_rate,
            src_attention_dropout_rate=config.src_attention_dropout_rate,
            input_layer="embed",
            use_output_layer=config.use_output_layer,
            pos_enc_class=PositionalEncoding,
            normalize_before=config.normalize_before,
            concat_after=config.concat_after,
        )
        self.ignore_id = config.ignore_id

    def forward(
        self,
        concepts: torch.Tensor, # (batch_size, concept_dim)
        captions_input_ids: torch.Tensor, # (batch_size, max_caption_length)
        captions_lengths: torch.Tensor, # (batch_size,)
    ) -> torch.Tensor:
        batch_size = concepts.size(0)
        device = concepts.device

        # Reshape concept_vector to be a sequence of length 1 for the decoder's 'memory'
        # (batch_size, 1, concept_dim)
        decoder_memory = concepts.unsqueeze(1)
        memory_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device) # Mask for the single concept token

        logits, _, _ = self.transformer_decoder(
            memory=decoder_memory,
            memory_mask=memory_mask,
            ys_in_pad=captions_input_ids,
            ys_in_lens=captions_lengths
        )
        return logits

    def generate_caption(
        self,
        concept_vector: torch.Tensor, # (1, concept_dim) for single inference
        max_len: int = 50,
        start_token_id: int = 1,
        end_token_id: int = 2,
    ) -> List[int]:
        self.eval()
        with torch.no_grad():
            batch_size = concept_vector.size(0)
            device = concept_vector.device

            # Reshape concept_vector to be a sequence of length 1 for the decoder's 'memory'
            # (batch_size, 1, concept_dim)
            decoder_memory = concept_vector.unsqueeze(1)
            memory_mask = torch.ones(batch_size, 1, 1, dtype=torch.bool, device=device) # Mask for the single concept token

            # Start with the start token
            generated_tokens = torch.tensor([[start_token_id]], device=device) # (batch_size, 1)

            for _ in range(max_len - 1): # -1 because we already have the start token
                # Create a mask for the target sequence (look-ahead mask)
                tgt_mask = subsequent_mask(generated_tokens.size(1)).unsqueeze(0).to(device)

                logits, _ = self.transformer_decoder.forward_one_step(
                    tgt=generated_tokens,
                    memory=decoder_memory,
                    tgt_mask=tgt_mask,
                    memory_mask=memory_mask
                )
                
                # Get the next token (greedy decoding)
                next_token_id = logits.argmax(dim=-1) # (batch_size,)
                generated_tokens = torch.cat([generated_tokens, next_token_id.unsqueeze(1)], dim=1)

                if next_token_id.item() == end_token_id: # Assuming batch_size is 1 for inference
                    break
            
            return generated_tokens.squeeze(0).tolist() # Convert to list of token IDs

class ConceptToImageVAE(nn.Module):
    def __init__(self, config: ConceptToImageVAEConfig = ConceptToImageVAEConfig()):
        super().__init__()

        if config.hidden_dims is None:
            hidden_dims = [512, 256, 128, 64] # Example hidden dimensions for upsampling
        else:
            hidden_dims = config.hidden_dims

        self.image_size = config.image_size
        self.num_upsampling_layers = len(hidden_dims) # Number of ConvTranspose2d layers

        # Calculate the initial spatial dimension (e.g., 4x4 for 64x64 image with 4 upsampling layers)
        self.initial_spatial_dim = self.image_size // (2 ** self.num_upsampling_layers)
        
        # Initial linear layer to project concept_dim to a flattened spatial representation
        self.fc = nn.Linear(config.concept_dim, hidden_dims[0] * (self.initial_spatial_dim ** 2))

        # Build convolutional transpose layers for upsampling
        modules = []
        in_channels = hidden_dims[0]
        
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.decoder_blocks = nn.Sequential(*modules)

        # Final convolutional layer to output the image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, config.output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Output images in [-1, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Project and reshape to initial spatial dimensions
        x = self.fc(z)
        x = x.view(-1, self.decoder_blocks[0][0].in_channels, self.initial_spatial_dim, self.initial_spatial_dim)
        
        x = self.decoder_blocks(x)
        x = self.final_layer(x)
        return x

class ImageGenerationDecoder(nn.Module):
    def __init__(
        self,
        config: ConceptToImageVAEConfig = ConceptToImageVAEConfig()
    ):
        super().__init__()
        self.vae_decoder = ConceptToImageVAE(
            config=config
        )

    def forward(self, concepts: torch.Tensor) -> List[Image.Image]:
        if not isinstance(concepts, torch.Tensor) or concepts.dim() != 2:
            raise ValueError("Input 'concepts' must be a 2D torch.Tensor (batch_size, concept_dim).")

        # Generate image tensor from concepts
        image_tensor = self.vae_decoder(concepts) # (batch_size, channels, height, width)

        # Convert tensor to PIL Images
        # Denormalize from [-1, 1] to [0, 255]
        image_tensor = (image_tensor + 1) / 2
        image_tensor = (image_tensor * 255).byte()

        pil_images = []
        for i in range(image_tensor.size(0)):
            # Permute from (C, H, W) to (H, W, C) for PIL
            img_np = image_tensor[i].permute(1, 2, 0).cpu().numpy()
            pil_images.append(Image.fromarray(img_np))
        
        return pil_images

    def generate_image(self, concept_vector: torch.Tensor) -> Image.Image:
        if not isinstance(concept_vector, torch.Tensor) or concept_vector.dim() != 2 or concept_vector.size(0) != 1:
            raise ValueError("Input 'concept_vector' must be a 2D torch.Tensor of shape (1, concept_dim).")
        
        # Call the forward method with a batch containing the single concept vector
        images = self.forward(concept_vector)
        return images[0]
