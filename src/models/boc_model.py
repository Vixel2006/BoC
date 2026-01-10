import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional

from .vit_encoder import ViTEncoder
from .text_encoder import TextEncoder
from .vae_decoder import VAEDecoder, SimpleVAEDecoder
from .text_decoder import TextDecoder
from .vq import VectorQuantizer


class BoCModel(nn.Module):
    """
    Bag of Concepts: Unified multimodal model.
    
    This model combines:
    - Image encoding (ViT → VQ)
    - Image decoding (VQ → VAE)
    - Text encoding (TextTransformer → VQ, shared codebook)
    - Text decoding (VQ → Transformer Decoder)
    - Cross-modal alignment via InfoNCE
    
    Args:
        # Image encoder
        image_size: Input image size (assumes square)
        patch_size: ViT patch size
        image_channels: Number of image channels (3 for RGB)
        
        # Text encoder
        vocab_size: Vocabulary size
        max_text_len: Maximum text sequence length
        
        # Shared VQ config
        num_embeddings: Codebook size
        embed_dim: Embedding dimension
        
        # Encoder architecture
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: MLP hidden dimension
        
        # Decoder architecture
        decoder_num_layers: Number of layers in decoders
        
        # VAE decoder
        use_simple_vae: If True, use SimpleVAEDecoder; else use full VAEDecoder
        vae_base_channels: Base channels for VAE decoder
        
        # VQ parameters
        vq_commitment_cost: Commitment loss weight
        vq_decay: EMA decay
        
        # Other
        dropout_rate: Dropout rate
    """
    # Image config
    image_size: int = 224
    patch_size: int = 16
    image_channels: int = 3
    
    # Text config
    vocab_size: int = 30000
    max_text_len: int = 128
    
    # Shared embedding config
    num_embeddings: int = 512
    embed_dim: int = 384
    
    # Encoder config
    num_heads: int = 6
    num_layers: int = 6
    mlp_dim: int = 1536
    
    # Decoder config
    decoder_num_layers: int = 4
    
    # VAE config
    use_simple_vae: bool = True
    vae_base_channels: int = 64
    
    # VQ config
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99
    
    # Other
    dropout_rate: float = 0.1
    
    def setup(self):
        """Initialize all model components."""
        # Image encoder
        self.image_encoder = ViTEncoder(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
            num_embeddings=self.num_embeddings,
            vq_commitment_cost=self.vq_commitment_cost,
            vq_decay=self.vq_decay,
        )
        
        # Text encoder (shares VQ codebook conceptually via same config)
        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            max_len=self.max_text_len,
            dropout_rate=self.dropout_rate,
            num_embeddings=self.num_embeddings,
            vq_commitment_cost=self.vq_commitment_cost,
            vq_decay=self.vq_decay,
        )
        
        # Image decoder
        if self.use_simple_vae:
            self.image_decoder = SimpleVAEDecoder(
                embed_dim=self.embed_dim,
                base_channels=self.vae_base_channels,
                output_channels=self.image_channels,
                num_upsample=4,  # 14x14 -> 224x224
            )
        else:
            self.image_decoder = VAEDecoder(
                embed_dim=self.embed_dim,
                hidden_dims=(256, 128, 64, 32),
                output_channels=self.image_channels,
                image_size=self.image_size,
                patch_size=self.patch_size,
            )
        
        # Text decoder
        self.text_decoder = TextDecoder(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.decoder_num_layers,
            mlp_dim=self.mlp_dim,
            max_len=self.max_text_len,
            dropout_rate=self.dropout_rate,
        )
    
    def encode_image(
        self, 
        images: jax.Array, 
        training: bool = False,
        return_all_tokens: bool = True
    ) -> Tuple[jax.Array, Dict]:
        """Encode images to VQ codes."""
        return self.image_encoder(images, training=training, return_all_tokens=return_all_tokens)
    
    def encode_text(
        self, 
        text: jax.Array, 
        training: bool = False,
        return_all_tokens: bool = False
    ) -> Tuple[jax.Array, Dict]:
        """Encode text to VQ codes."""
        return self.text_encoder(text, training=training, return_all_tokens=return_all_tokens)
    
    def decode_image(
        self, 
        quantized: jax.Array, 
        training: bool = False
    ) -> jax.Array:
        """Decode VQ codes to images."""
        # quantized shape: (B, num_patches, embed_dim)
        # Reshape to spatial for VAE decoder
        batch_size = quantized.shape[0]
        num_patches = quantized.shape[1]
        spatial_size = int(num_patches ** 0.5)
        
        quantized_spatial = quantized.reshape(
            batch_size, spatial_size, spatial_size, self.embed_dim
        )
        
        return self.image_decoder(quantized_spatial, training=training)
    
    def decode_text(
        self, 
        text_tokens: jax.Array,
        quantized: jax.Array, 
        training: bool = False
    ) -> jax.Array:
        """Decode VQ codes to text (teacher forcing)."""
        # quantized can be (B, num_patches, embed_dim) or (B, embed_dim)
        # text_tokens: (B, seq_len) - target tokens for teacher forcing
        return self.text_decoder(text_tokens, context=quantized, training=training)
    
    def image_autoencoder(
        self, 
        images: jax.Array, 
        training: bool = False
    ) -> Tuple[jax.Array, Dict]:
        """
        Full image autoencoder: Image → VQ → Image
        
        Returns:
            reconstructed: Reconstructed images
            info: Dictionary with losses and metrics
        """
        # Encode
        quantized, enc_info = self.encode_image(images, training=training, return_all_tokens=True)
        
        # Decode
        reconstructed = self.decode_image(quantized, training=training)
        
        return reconstructed, enc_info
    
    def text_autoencoder(
        self,
        text_tokens: jax.Array,
        training: bool = False
    ) -> Tuple[jax.Array, Dict]:
        """
        Full text autoencoder: Text → VQ → Text
        
        Args:
            text_tokens: (B, seq_len) token IDs
            
        Returns:
            logits: (B, seq_len, vocab_size)
            info: Dictionary with losses and metrics
        """
        # Encode to VQ
        quantized, enc_info = self.encode_text(text_tokens, training=training, return_all_tokens=True)
        
        # Decode (teacher forcing)
        logits = self.decode_text(text_tokens, quantized, training=training)
        
        return logits, enc_info
    
    def text_to_image(
        self,
        text_tokens: jax.Array,
        training: bool = False
    ) -> jax.Array:
        """Generate images from text."""
        # Encode text to VQ
        quantized, _ = self.encode_text(text_tokens, training=training, return_all_tokens=True)
        
        # Decode to image
        images = self.decode_image(quantized, training=training)
        
        return images
    
    def image_to_text(
        self,
        images: jax.Array,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[jax.Array] = None
    ) -> jax.Array:
        """Generate text from images."""
        # Encode image to VQ
        quantized, _ = self.encode_image(images, training=False, return_all_tokens=True)
        
        # Generate text autoregressively
        text = self.text_decoder.generate(
            context=quantized,
            start_token=start_token,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            rng=rng
        )
        
        return text
    
    def __call__(
        self,
        images: Optional[jax.Array] = None,
        text_tokens: Optional[jax.Array] = None,
        training: bool = False,
        mode: str = 'both'
    ) -> Dict[str, any]:
        """
        Forward pass supporting different training modes.
        
        Args:
            images: (B, H, W, C)
            text_tokens: (B, seq_len)
            training: Training mode flag
            mode: 'image_ae', 'text_ae', 'both', or 'alignment'
            
        Returns:
            Dictionary with outputs and losses depending on mode
        """
        outputs = {}
        
        if mode in ['image_ae', 'both'] and images is not None:
            recon_images, img_info = self.image_autoencoder(images, training=training)
            outputs['reconstructed_images'] = recon_images
            outputs['image_vq_info'] = img_info
        
        if mode in ['text_ae', 'both'] and text_tokens is not None:
            text_logits, text_info = self.text_autoencoder(text_tokens, training=training)
            outputs['text_logits'] = text_logits
            outputs['text_vq_info'] = text_info
        
        if mode == 'alignment' and images is not None and text_tokens is not None:
            # Encode both to pooled representations for contrastive learning
            img_emb, img_info = self.encode_image(images, training=training, return_all_tokens=False)
            text_emb, text_info = self.encode_text(text_tokens, training=training, return_all_tokens=False)
            
            outputs['image_embeddings'] = img_emb
            outputs['text_embeddings'] = text_emb
            outputs['image_vq_info'] = img_info
            outputs['text_vq_info'] = text_info
        
        return outputs
