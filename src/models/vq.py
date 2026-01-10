import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Dict
import functools

class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with codebook collapse mitigation.
    
    Features:
    - EMA-based codebook updates for stable training
    - Dead code detection and reset
    - Entropy regularization for uniform code usage
    - Configurable commitment loss
    
    Args:
        num_embeddings: Size of the codebook (number of discrete codes)
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss (encoder commitment to codes)
        decay: EMA decay rate for codebook updates (0.99 recommended)
        epsilon: Small constant for numerical stability
        use_entropy_loss: Whether to use entropy regularization
        entropy_loss_weight: Weight for entropy loss
        code_reset_threshold: Usage threshold below which codes are reset (e.g., 0.01)
    """
    num_embeddings: int
    embedding_dim: int
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    use_entropy_loss: bool = True
    entropy_loss_weight: float = 0.1
    code_reset_threshold: float = 0.01
    
    def setup(self):
        # Initialize codebook embeddings
        self.embedding = self.param(
            'embedding',
            nn.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='uniform'),
            (self.num_embeddings, self.embedding_dim)
        )
        
    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """
        Forward pass through VQ layer.
        
        Args:
            x: Input tensor of shape (Batch, Seq_Len, Embed_Dim) or (Batch, H, W, Embed_Dim)
            training: Whether in training mode
            
        Returns:
            quantized: Quantized representation (same shape as x)
            losses: Dictionary containing:
                - vq_loss: Total VQ loss (commitment + codebook loss)
                - commitment_loss: Encoder commitment loss
                - codebook_loss: Codebook update loss (always 0 for EMA)
                - perplexity: Codebook usage perplexity
                - entropy_loss: Entropy regularization loss (optional)
                - encoding_indices: Discrete code indices
        """
        # Flatten spatial dimensions if present
        input_shape = x.shape
        flat_x = x.reshape(-1, self.embedding_dim)  # (B*L, D) or (B*H*W, D)
        
        # Get the codebook
        codebook = self.embedding
        
        # Calculate distances from each input to all codebook vectors
        # Using: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        distances = (
            jnp.sum(flat_x ** 2, axis=1, keepdims=True) +
            jnp.sum(codebook ** 2, axis=1) -
            2 * jnp.matmul(flat_x, codebook.T)
        )  # (B*L, num_embeddings)
        
        # Find nearest codebook entry for each input
        encoding_indices = jnp.argmin(distances, axis=1)  # (B*L,)
        
        # Get quantized values
        quantized_flat = codebook[encoding_indices]  # (B*L, D)
        quantized = quantized_flat.reshape(input_shape)  # Restore original shape
        
        # Commitment loss: encourages encoder to commit to codebook entries
        commitment_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
        
        # For EMA updates, we don't use explicit codebook loss
        # The codebook is updated through EMA in a separate training step
        # Here we use stop_gradient to prevent gradient flow to encoder
        codebook_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
        
        # Straight-through estimator: copy gradients from quantized to input
        quantized = x + jax.lax.stop_gradient(quantized - x)
        
        # Calculate codebook usage statistics
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings)  # (B*L, num_embeddings)
        avg_probs = jnp.mean(encodings, axis=0)  # (num_embeddings,)
        
        # Perplexity: measure of codebook usage diversity
        # Higher is better (max = num_embeddings when all codes used equally)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + self.epsilon)))
        
        # Entropy loss: encourages uniform code distribution
        entropy_loss = 0.0
        if self.use_entropy_loss:
            # Negative entropy (we want to minimize this to maximize entropy)
            entropy_loss = jnp.sum(avg_probs * jnp.log(avg_probs + self.epsilon))
        
        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        if self.use_entropy_loss:
            vq_loss = vq_loss + self.entropy_loss_weight * entropy_loss
        
        # Return quantized representation and loss dictionary
        losses = {
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'perplexity': perplexity,
            'entropy_loss': entropy_loss,
            'encoding_indices': encoding_indices.reshape(input_shape[:-1]),  # (B, L) or (B, H, W)
            'avg_probs': avg_probs,  # For monitoring code usage
        }
        
        return quantized, losses
    
    def get_codebook_entry(self, indices: jax.Array) -> jax.Array:
        """
        Retrieve codebook vectors for given indices.
        
        Args:
            indices: Tensor of shape (Batch, ...) containing code indices
            
        Returns:
            Embeddings of shape (Batch, ..., Embed_Dim)
        """
        codebook = self.embedding
        return codebook[indices]


def update_codebook_ema(
    codebook: jax.Array,
    ema_cluster_size: jax.Array,
    ema_w: jax.Array,
    encodings: jax.Array,
    flat_input: jax.Array,
    decay: float,
    epsilon: float,
    reset_threshold: float = 0.01
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Update codebook using Exponential Moving Average (EMA).
    
    This should be called as a separate training step, not during the forward pass.
    
    Args:
        codebook: Current codebook of shape (num_embeddings, embedding_dim)
        ema_cluster_size: EMA of cluster sizes, shape (num_embeddings,)
        ema_w: EMA of cluster sums, shape (num_embeddings, embedding_dim)
        encodings: One-hot encodings, shape (batch_size, num_embeddings)
        flat_input: Flattened input, shape (batch_size, embedding_dim)
        decay: EMA decay rate
        epsilon: Small constant for stability
        reset_threshold: Threshold for resetting dead codes
        
    Returns:
        updated_codebook: Updated codebook
        updated_ema_cluster_size: Updated cluster sizes
        updated_ema_w: Updated cluster sums
    """
    # Update cluster size EMA
    cluster_size = jnp.sum(encodings, axis=0)  # (num_embeddings,)
    updated_ema_cluster_size = decay * ema_cluster_size + (1 - decay) * cluster_size
    
    # Update embedding sum EMA
    dw = jnp.matmul(encodings.T, flat_input)  # (num_embeddings, embedding_dim)
    updated_ema_w = decay * ema_w + (1 - decay) * dw
    
    # Normalize to get updated embeddings
    n = jnp.sum(updated_ema_cluster_size)
    normalized_cluster_size = (
        (updated_ema_cluster_size + epsilon) /
        (n + codebook.shape[0] * epsilon)
    )
    updated_codebook = updated_ema_w / normalized_cluster_size[:, None]
    
    # Reset dead codes (codes used less than threshold)
    # Randomly reinitialize from the input batch
    usage_ratio = updated_ema_cluster_size / (n + epsilon)
    dead_mask = usage_ratio < reset_threshold
    
    # For dead codes, sample random vectors from the current batch
    num_dead = jnp.sum(dead_mask)
    if num_dead > 0:
        # Randomly select vectors from batch to replace dead codes
        batch_size = flat_input.shape[0]
        random_indices = jax.random.randint(
            jax.random.PRNGKey(0), 
            (codebook.shape[0],), 
            0, 
            batch_size
        )
        random_vectors = flat_input[random_indices]
        
        # Replace dead codes
        updated_codebook = jnp.where(
            dead_mask[:, None],
            random_vectors,
            updated_codebook
        )
        
        # Reset EMA statistics for dead codes
        updated_ema_cluster_size = jnp.where(
            dead_mask,
            jnp.ones_like(updated_ema_cluster_size),
            updated_ema_cluster_size
        )
        updated_ema_w = jnp.where(
            dead_mask[:, None],
            random_vectors,
            updated_ema_w
        )
    
    return updated_codebook, updated_ema_cluster_size, updated_ema_w
