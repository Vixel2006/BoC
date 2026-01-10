import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from .transformer import DecoderBlock

class TextDecoder(nn.Module):
    """
    Transformer decoder for autoregressive text generation from VQ embeddings.
    
    Used to reconstruct text from concept codes and for image-to-text generation.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Dimension of embeddings (should match VQ embedding dim)
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        mlp_dim: Hidden dimension in MLP blocks
        max_len: Maximum sequence length
        dropout_rate: Dropout rate
    """
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    mlp_dim: int
    max_len: int = 512
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        context: Optional[jax.Array] = None,
        training: bool = False
    ) -> jax.Array:
        """
        Args:
            x: Input token indices (B, seq_len)
            context: VQ embeddings to condition on (B, num_codes, embed_dim)
            training: Whether in training mode
            
        Returns:
            Logits of shape (B, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(x)
        
        # Positional embeddings
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, self.max_len, self.embed_dim)
        )
        x = x + pos_embedding[:, :seq_len, :]
        
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        # Decoder layers with causal self-attention and cross-attention to VQ codes
        for i in range(self.num_layers):
            x = DecoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'decoder_block_{i}'
            )(x, context=context, training=training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Output projection to vocabulary
        x = nn.Dense(features=self.vocab_size)(x)
        
        return x
    
    def generate(
        self,
        context: jax.Array,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        rng: Optional[jax.Array] = None
    ) -> jax.Array:
        """
        Autoregressive generation from VQ embeddings.
        
        Args:
            context: VQ embeddings (B, num_codes, embed_dim)
            start_token: Token ID to start generation
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, sample from top-k tokens only
            rng: Random key for sampling
            
        Returns:
            Generated token sequence (B, max_length)
        """
        batch_size = context.shape[0]
        
        # Initialize with start token
        tokens = jnp.full((batch_size, 1), start_token, dtype=jnp.int32)
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self(tokens, context=context, training=False)  # (B, curr_len, vocab_size)
            
            # Take logits for last position
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            
            # Top-k filtering if specified
            if top_k is not None:
                # Get top-k values and indices
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                
                # Create mask for top-k
                mask = jnp.full_like(next_token_logits, -jnp.inf)
                mask = mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(0.0)
                next_token_logits = next_token_logits + mask
            
            # Sample next token
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_token_logits, axis=-1)
            next_token = next_token[:, None]  # (B, 1)
            
            # Append to sequence
            tokens = jnp.concatenate([tokens, next_token], axis=1)
        
        return tokens
