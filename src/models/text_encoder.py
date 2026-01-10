import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Tuple
from .transformer import MLP, EncoderBlock
from .vq import VectorQuantizer

class TextEncoder(nn.Module):
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
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        Encode text to VQ codes.
        
        Args:
            x: Token indices (B, seq_len)
            training: Whether in training mode
            return_all_tokens: If True, return all token embeddings; if False, return pooled
            
        Returns:
            quantized: Quantized embeddings (B, seq_len, embed_dim) or (B, embed_dim)
            info: Dictionary with VQ losses and encoding indices
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
        
        # 6. Vector Quantization
        vq = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embed_dim,
            commitment_cost=self.vq_commitment_cost,
            decay=self.vq_decay,
            name='vq'
        )
        quantized, vq_losses = vq(x, training=training)
        
        # 7. Global Average Pooling if not returning all tokens
        if not return_all_tokens:
            # Masked average pooling (ignore padding)
            mask_expanded = mask.reshape(b, seq_len, 1)
            quantized_sum = jnp.sum(quantized * mask_expanded, axis=1)
            scale = jnp.sum(mask_expanded, axis=1)
            scale = jnp.maximum(scale, 1e-9)
            quantized = quantized_sum / scale  # (B, embed_dim)

        # Return quantized embeddings and VQ info
        info = {
            'vq_loss': vq_losses['vq_loss'],
            'commitment_loss': vq_losses['commitment_loss'],
            'codebook_loss': vq_losses['codebook_loss'],
            'perplexity': vq_losses['perplexity'],
            'entropy_loss': vq_losses['entropy_loss'],
            'encoding_indices': vq_losses['encoding_indices'],
            'avg_probs': vq_losses['avg_probs'],
        }
        
        return quantized, info


# Keep original TextTransformer for compatibility
class TextTransformer(nn.Module):
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_len: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (Batch, Seq_Len) -- integers (token indices)
        b, seq_len = x.shape

        # 1. Create Attention Mask for Padding (assuming 0 is padding)
        # Shape: (Batch, 1, 1, Seq_Len) for broadcasting over Heads and Query position
        mask = jnp.where(x > 0, 1, 0).astype(jnp.int32)
        mask = mask[:, None, None, :]

        # 2. Token Embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)

        # 3. Positional Embeddings
        # Learnable position embeddings
        pos_embedding = self.param(
            'pos_embedding', 
            nn.initializers.normal(stddev=0.02), 
            (1, self.max_len, self.embed_dim)
        )
        # Slice to current sequence length
        x = x + pos_embedding[:, :seq_len, :]
        
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

        # 4. Transformer Encoder Layers
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads, 
                mlp_dim=self.mlp_dim, 
                dropout_rate=self.dropout_rate,
                name=f'text_encoder_block_{i}'
            )(x, mask=mask, training=training)

        # 5. Global Average Pooling (Masked)
        # We must ignore padding tokens in the average calculation
        # mask shape is (B, 1, 1, L), reshape to (B, L, 1) to multiply with x
        mask_expanded = mask.reshape(b, seq_len, 1)
        
        # Sum of features ignoring padding
        x_sum = jnp.sum(x * mask_expanded, axis=1)
        
        # Count of non-padding tokens
        scale = jnp.sum(mask_expanded, axis=1)
        scale = jnp.maximum(scale, 1e-9) # avoid div by zero
        
        x = x_sum / scale # (Batch, Embed_Dim)

        # Final Norm
        x = nn.LayerNorm()(x)
        
        return x
