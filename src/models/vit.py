import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Tuple, Optional
from .transformer import MLP, EncoderBlock
from .vq import VectorQuantizer

class ViT(nn.Module):
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
    ) -> jax.Array:
        """
        Encode image to VQ codes.
        
        Args:
            x: Images (B, H, W, C)
            training: Whether in training mode
            return_all_tokens: If True, return all patch tokens; if False, return pooled
            
        Returns:
            x: final hidden state (B, num_patches, embed_dim)
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
        
        return x
