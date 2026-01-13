import jax
import jax.numpy as jnp
from typing import Dict, Tuple

def info_nce_loss(
    image_embeddings: jax.Array,
    text_embeddings: jax.Array,
    temperature: float = 0.07,
    symmetric: bool = True
) -> Dict[str, jax.Array]:
    """
    InfoNCE contrastive loss for aligning image and text embeddings.
    
    This loss encourages corresponding image-text pairs to have high similarity
    while pushing non-corresponding pairs apart.
    
    Args:
        image_embeddings: Image embeddings (B, embed_dim)
        text_embeddings: Text embeddings (B, embed_dim)
        temperature: Temperature parameter for scaling similarities
        symmetric: If True, compute loss in both directions and average
        
    Returns:
        Dictionary containing:
            - loss: Total InfoNCE loss
            - image_to_text_loss: Loss for image→text direction
            - text_to_image_loss: Loss for text→image direction
            - logits: Similarity matrix (for monitoring)
    """
    batch_size = image_embeddings.shape[0]
    
    # Normalize embeddings to unit sphere
    image_embeddings = image_embeddings / (jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True) + 1e-8)
    text_embeddings = text_embeddings / (jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity matrix: (B, B)
    # logits[i, j] = similarity between image i and text j
    logits = jnp.matmul(image_embeddings, text_embeddings.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = jnp.arange(batch_size)
    
    # Image-to-text loss: for each image, find matching text
    image_to_text_loss = jnp.mean(
        -jax.nn.log_softmax(logits, axis=1)[jnp.arange(batch_size), labels]
    )
    
    # Text-to-image loss: for each text, find matching image
    text_to_image_loss = jnp.mean(
        -jax.nn.log_softmax(logits, axis=0)[labels, jnp.arange(batch_size)]
    )
    
    if symmetric:
        loss = (image_to_text_loss + text_to_image_loss) / 2
    else:
        loss = image_to_text_loss
    
    return {
        'loss': loss,
        'image_to_text_loss': image_to_text_loss,
        'text_to_image_loss': text_to_image_loss,
        'logits': logits,
    }


def reconstruction_loss_image(
    reconstructed: jax.Array,
    target: jax.Array,
    loss_type: str = 'mse'
) -> jax.Array:
    """
    Reconstruction loss for images.
    
    Args:
        reconstructed: Reconstructed images (B, H, W, C)
        target: Target images (B, H, W, C)
        loss_type: Type of loss ('mse' or 'l1')
        
    Returns:
        Scalar reconstruction loss
    """
    if loss_type == 'mse':
        return jnp.mean((reconstructed - target) ** 2)
    elif loss_type == 'l1':
        return jnp.mean(jnp.abs(reconstructed - target))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def reconstruction_loss_text(
    logits: jax.Array,
    targets: jax.Array,
    pad_token_id: int = 0
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Cross-entropy reconstruction loss for text with padding handling.
    
    Args:
        logits: Predicted logits (B, seq_len, vocab_size)
        targets: Target token IDs (B, seq_len)
        pad_token_id: ID of padding token to ignore in loss
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Create mask for non-padding tokens
    mask = (targets != pad_token_id).astype(jnp.float32)
    
    # Compute cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Gather log probabilities for target tokens
    batch_size, seq_len, vocab_size = logits.shape
    batch_indices = jnp.arange(batch_size)[:, None]
    seq_indices = jnp.arange(seq_len)[None, :]
    target_log_probs = log_probs[batch_indices, seq_indices, targets]
    
    # Apply mask and compute mean loss
    masked_loss = -target_log_probs * mask
    loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-8)
    
    # Compute perplexity
    perplexity = jnp.exp(loss)
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(jnp.float32) * mask
    accuracy = jnp.sum(correct) / (jnp.sum(mask) + 1e-8)
    
    metrics = {
        'perplexity': perplexity,
        'accuracy': accuracy,
    }
    
    return loss, metrics


def compute_codebook_metrics(
    avg_probs: jax.Array,
    num_embeddings: int,
    epsilon: float = 1e-10
) -> Dict[str, jax.Array]:
    """
    Compute metrics for monitoring codebook usage.
    
    Args:
        avg_probs: Average probability of each code (num_embeddings,)
        num_embeddings: Total number of codes in codebook
        epsilon: Small constant for numerical stability
        
    Returns:
        Dictionary with metrics:
            - perplexity: Codebook usage perplexity (higher is better)
            - active_codes: Number of codes used above threshold
            - entropy: Entropy of code distribution (higher is better)
            - max_prob: Maximum probability (lower is better for uniform usage)
    """
    # Perplexity: exp(-sum(p * log(p)))
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + epsilon)))
    
    # Active codes: count how many codes are used above threshold
    threshold = 1.0 / (num_embeddings * 2)  # Half of uniform probability
    active_codes = jnp.sum(avg_probs > threshold)
    
    # Entropy
    entropy = -jnp.sum(avg_probs * jnp.log(avg_probs + epsilon))
    
    # Max probability (should be close to 1/num_embeddings for uniform usage)
    max_prob = jnp.max(avg_probs)
    
    return {
        'codebook_perplexity': perplexity,
        'codebook_active_codes': active_codes,
        'codebook_entropy': entropy,
        'codebook_max_prob': max_prob,
        'codebook_usage_ratio': active_codes / num_embeddings,
    }


def total_vq_autoencoder_loss(
    reconstruction_loss: jax.Array,
    vq_loss: jax.Array,
    commitment_loss: jax.Array,
    entropy_loss: jax.Array = 0.0,
    reconstruction_weight: float = 1.0,
    vq_weight: float = 1.0,
    commitment_weight: float = 0.25,
    entropy_weight: float = 0.1
) -> Dict[str, jax.Array]:
    """
    Combined loss for VQ autoencoder (image or text).
    
    Args:
        reconstruction_loss: Reconstruction loss (MSE for images, CE for text)
        vq_loss: VQ codebook loss
        commitment_loss: Commitment loss from VQ
        entropy_loss: Entropy regularization loss
        reconstruction_weight: Weight for reconstruction loss
        vq_weight: Weight for VQ loss
        commitment_weight: Weight for commitment loss
        entropy_weight: Weight for entropy loss
        
    Returns:
        Dictionary with total loss and individual components
    """
    total = (
        reconstruction_weight * reconstruction_loss +
        vq_weight * vq_loss +
        commitment_weight * commitment_loss +
        entropy_weight * entropy_loss
    )
    
    return {
        'total_loss': total,
        'reconstruction_loss': reconstruction_loss,
        'vq_loss': vq_loss,
        'commitment_loss': commitment_loss,
        'entropy_loss': entropy_loss,
    }
