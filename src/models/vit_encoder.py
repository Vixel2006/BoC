import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional
from .transformer import MLP, EncoderBlock
from .vq import VectorQuantizer

class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder with Vector Quantization.
    
    This is the image encoder for BoC that maps images to discrete concept codes.
    
    Args:
        patch_size: Size of image patches
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: Hidden dimension in MLP blocks
        dropout_rate: Dropout rate
        num_embeddings: Size of VQ codebook  
        vq_commitment_cost: VQ commitment loss weight
        vq_decay: EMA decay for VQ updates
    """
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    dropout_rate: float = 0.1
    num_embeddings: int = 512
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99

    @nn.compact
    def __call__(
        self, 
        x: jax.Array, 
        training: bool = False,
        return_all_tokens: bool = True
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        Encode image to VQ codes.
        
        Args:
            x: Images (B, H, W, C)
            training: Whether in training mode
            return_all_tokens: If True, return all patch tokens; if False, return pooled
            
        Returns:
            quantized: Quantized embeddings (B, num_patches, embed_dim) or (B, embed_dim)
            info: Dictionary with VQ losses and encoding indices
        """
        # x shape: (Batch, Height, Width, Channels)
        b, h, w, c = x.shape
        
        # 1. Patch Embedding
        # Output: (B, H', W', Embed_Dim)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        
        # Flatten: (B, Seq_Len, Embed_Dim)
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        x = x.reshape(b, -1, self.embed_dim)
        
        # 2. Add Positional Embeddings
        seq_len = x.shape[1]
        pos_embedding = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.02), 
            (1, seq_len, self.embed_dim)
        )
        
        x = x + pos_embedding
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # 3. Transformer Encoder Layers
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads, 
                mlp_dim=self.mlp_dim, 
                dropout_rate=self.dropout_rate,
                name=f'encoder_block_{i}'
            )(x, training=training)

        # 4. Layer Norm before VQ
        x = nn.LayerNorm()(x)
        
        # 5. Vector Quantization
        vq = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embed_dim,
            commitment_cost=self.vq_commitment_cost,
            decay=self.vq_decay,
            name='vq'
        )
        quantized, vq_losses = vq(x, training=training)
        
        # 6. Optional pooling
        if not return_all_tokens:
            quantized = jnp.mean(quantized, axis=1)  # (B, embed_dim)
        
        # Return quantized embeddings and VQ info
        info = {
            'vq_loss': vq_losses['vq_loss'],
            'commitment_loss': vq_losses['commitment_loss'],
            'codebook_loss': vq_losses['codebook_loss'],
            'perplexity': vq_losses['perplexity'],
            'entropy_loss': vq_losses['entropy_loss'],
            'encoding_indices': vq_losses['encoding_indices'],  # (B, num_patches_h, num_patches_w)
            'avg_probs': vq_losses['avg_probs'],
        }
        
        return quantized, info


# Keep original ViT for compatibility
class ViT(nn.Module):
    patch_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    num_classes: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (Batch, Height, Width, Channels)
        b, h, w, c = x.shape
        
        # 1. Patch Embedding
        # Output: (B, H', W', Embed_Dim)
        x = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        
        # Flatten: (B, Seq_Len, Embed_Dim)
        x = x.reshape(b, -1, self.embed_dim)
        
        # --- REMOVED CLS TOKEN CREATION AND CONCATENATION ---
        
        # 2. Add Positional Embeddings
        # Seq_Len is now just the number of patches (H*W / P*P)
        seq_len = x.shape[1]
        pos_embedding = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.02), 
            (1, seq_len, self.embed_dim)
        )
        
        x = x + pos_embedding
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # 3. Transformer Encoder Layers
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads, 
                mlp_dim=self.mlp_dim, 
                dropout_rate=self.dropout_rate,
                name=f'encoder_block_{i}'
            )(x, training=training)

        # 4. Pooling & Classification Head
        
        # Instead of taking x[:, 0], we take the mean over the sequence dimension (axis 1)
        # Shape: (Batch, Seq_Len, Embed_Dim) -> (Batch, Embed_Dim)
        x = jnp.mean(x, axis=1)
        
        # Final Norm
        x = nn.LayerNorm()(x)
        
        # Output projection
        x = nn.Dense(features=self.num_classes)(x)
        
        return x
