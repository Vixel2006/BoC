import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class MLP(nn.Module):
    hidden_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False) -> jax.Array:
        actual_dim = x.shape[-1]
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(features=actual_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x

class EncoderBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask: Optional[jax.Array] = None, training: bool = False) -> jax.Array:
        # Attention Block
        y = nn.LayerNorm()(x)
        
        # We pass the mask here. 
        # Flax MHA expects mask shape: (batch, num_heads, q_len, kv_len)
        # or broadcastable like (batch, 1, 1, kv_len)
        y = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1],
            dropout_rate=self.dropout_rate
        )(y, y, mask=mask, deterministic=not training)
        
        x = x + y

        # MLP Block
        y = nn.LayerNorm()(x)
        y = MLP(hidden_dim=self.mlp_dim, dropout_rate=self.dropout_rate)(y, training=training)

        return x + y

class DecoderBlock(nn.Module):
    """
    Transformer decoder block with causal self-attention and cross-attention.
    Used for autoregressive text generation conditioned on VQ embeddings.
    """
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self, 
        x: jax.Array,
        context: Optional[jax.Array] = None,
        causal_mask: Optional[jax.Array] = None,
        training: bool = False
    ) -> jax.Array:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            context: Context from encoder for cross-attention (batch, context_len, dim)
            causal_mask: Causal mask for autoregressive generation
            training: Whether in training mode
        """
        # Causal Self-Attention Block
        y = nn.LayerNorm()(x)
        
        # Create causal mask if not provided
        if causal_mask is None:
            seq_len = x.shape[1]
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
            causal_mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)
        
        y = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1],
            dropout_rate=self.dropout_rate
        )(y, y, mask=causal_mask, deterministic=not training)
        
        x = x + y
        
        # Cross-Attention Block (if context provided)
        if context is not None:
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=x.shape[-1],
                dropout_rate=self.dropout_rate
            )(y, context, deterministic=not training)
            x = x + y
        
        # MLP Block
        y = nn.LayerNorm()(x)
        y = MLP(hidden_dim=self.mlp_dim, dropout_rate=self.dropout_rate)(y, training=training)
        
        return x + y
