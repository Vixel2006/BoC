import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Tuple
from .transformer import MLP, EncoderBlock
from .vq import VectorQuantizer

class TextFormer(nn.Module):
    """
    Text Transformer encoder with Vector Quantization.
    
    This encodes text to the same VQ space as images for cross-modal alignment.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension (should match image encoder)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: Hidden dimension in MLP blocks
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
        num_embeddings: Size of VQ codebook (should match image encoder)
        vq_commitment_cost: VQ commitment loss weight
        vq_decay: EMA decay for VQ updates
    """
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_len: int
    dropout_rate: float = 0.1
    num_embeddings: int = 512
    vq_commitment_cost: float = 0.25
    vq_decay: float = 0.99

    @nn.compact
    def __call__(
        self, 
        x: jax.Array, 
        training: bool = False,
        return_all_tokens: bool = False
    ) -> jax.Array:
        """
        Encode text to VQ codes.
        
        Args:
            x: Token indices (B, seq_len)
            training: Whether in training mode
            return_all_tokens: If True, return all token embeddings; if False, return pooled
            
        Returns:
            x: final hidden state (B, seq_len, embed_dim)
        """
        # x shape: (Batch, Seq_Len) -- integers (token indices)
        b, seq_len = x.shape

        # 1. Create Attention Mask for Padding (assuming 0 is padding)
        # Shape: (Batch, 1, 1, Seq_Len) for broadcasting over Heads and Query position
        mask = jnp.where(x > 0, 1, 0).astype(jnp.int32)
        padding_mask = mask[:, None, None, :]

        # 2. Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)

        # 3. Positional Embeddings
        pos_embedding = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.02), 
            (1, self.max_len, self.embed_dim)
        )
        x = x + pos_embedding[:, :seq_len, :]
        
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # 4. Transformer Encoder Layers
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads, 
                mlp_dim=self.mlp_dim, 
                dropout_rate=self.dropout_rate,
                name=f'text_encoder_block_{i}'
            )(x, mask=padding_mask, training=training)

        # 5. Layer Norm before VQ
        x = nn.LayerNorm()(x)
        
        return x
