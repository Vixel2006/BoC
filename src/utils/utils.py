import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [-1, 1] range.
    
    Args:
        image: Image array in [0, 255] or [0, 1] range
        
    Returns:
        Normalized image in [-1, 1]
    """
    if image.max() > 1.0:
        image = image / 255.0
    return image * 2.0 - 1.0


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [-1, 1] to [0, 255].
    
    Args:
        image: Image in [-1, 1] range
        
    Returns:
        Image in [0, 255] range as uint8
    """
    image = (image + 1.0) / 2.0
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def load_image(path: str, size: int = 224) -> np.ndarray:
    """
    Load and preprocess image.
    
    Args:
        path: Path to image file
        size: Target size (square)
        
    Returns:
        Preprocessed image array (H, W, C)
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    img = np.array(img)
    return normalize_image(img)


def save_image(image: np.ndarray, path: str):
    """
    Save image to file.
    
    Args:
        image: Image in [-1, 1] range
        path: Output path
    """
    img = denormalize_image(image)
    Image.fromarray(img).save(path)


def visualize_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize original and reconstructed images side by side.
    
    Args:
        original: Original images (B, H, W, C) in [-1, 1]
        reconstructed: Reconstructed images (B, H, W, C) in [-1, 1]
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure
    """
    batch_size = min(8, original.shape[0])
    
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))
    
    for i in range(batch_size):
        # Original
        axes[0, i].imshow(denormalize_image(original[i]))
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        axes[1, i].imshow(denormalize_image(reconstructed[i]))
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def visualize_codebook_usage(
    avg_probs: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize codebook usage distribution.
    
    Args:
        avg_probs: Average probability of each code (num_embeddings,)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].bar(range(len(avg_probs)), avg_probs)
    axes[0].set_xlabel('Code Index')
    axes[0].set_ylabel('Usage Probability')
    axes[0].set_title('Codebook Usage Distribution')
    axes[0].grid(alpha=0.3)
    
    # Sorted usage
    sorted_probs = np.sort(avg_probs)[::-1]
    axes[1].plot(sorted_probs)
    axes[1].set_xlabel('Code Rank')
    axes[1].set_ylabel('Usage Probability')
    axes[1].set_title('Sorted Codebook Usage')
    axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')
    
    # Add statistics
    num_active = np.sum(avg_probs > (1.0 / (len(avg_probs) * 2)))
    perplexity = np.exp(-np.sum(avg_probs * np.log(avg_probs + 1e-10)))
    
    fig.suptitle(
        f'Active Codes: {num_active}/{len(avg_probs)} | Perplexity: {perplexity:.1f}',
        fontsize=12
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def compute_psnr(
    original: jax.Array,
    reconstructed: jax.Array,
    max_val: float = 2.0  # For [-1, 1] range
) -> jax.Array:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        max_val: Maximum possible value (2.0 for [-1, 1] range)
        
    Returns:
        PSNR value
    """
    mse = jnp.mean((original - reconstructed) ** 2)
    psnr = 20 * jnp.log10(max_val) - 10 * jnp.log10(mse)
    return psnr


def create_dummy_batch(
    batch_size: int = 4,
    image_size: int = 224,
    max_text_len: int = 128,
    vocab_size: int = 30000,
    rng_key: Optional[jax.Array] = None
) -> dict:
    """
    Create dummy batch for testing.
    
    Args:
        batch_size: Batch size
        image_size: Image size
        max_text_len: Maximum text length
        vocab_size: Vocabulary size
        rng_key: Random key
        
    Returns:
        Dictionary with 'images' and 'text_tokens'
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    
    img_rng, text_rng = jax.random.split(rng_key)
    
    # Random images in [-1, 1]
    images = jax.random.uniform(img_rng, (batch_size, image_size, image_size, 3))
    images = images * 2 - 1
    
    # Random text tokens (with some padding)
    text_tokens = jax.random.randint(
        text_rng,
        (batch_size, max_text_len),
        1,  # Start from 1 (0 is padding)
        vocab_size
    )
    
    # Add some padding at the end
    for i in range(batch_size):
        pad_start = jax.random.randint(text_rng, (), max_text_len // 2, max_text_len)
        text_tokens = text_tokens.at[i, pad_start:].set(0)
    
    return {
        'images': images,
        'text_tokens': text_tokens
    }


def print_model_summary(params: dict):
    """
    Print summary of model parameters.
    
    Args:
        params: Model parameters dictionary
    """
    def count_params(params_dict, prefix=''):
        total = 0
        for key, value in params_dict.items():
            if isinstance(value, dict):
                count = count_params(value, prefix=f"{prefix}{key}.")
                total += count
            else:
                count = np.prod(value.shape)
                total += count
                print(f"{prefix}{key}: {value.shape} ({count:,} params)")
        return total
    
    print("\n" + "="*60)
    print("Model Parameter Summary")
    print("="*60)
    total_params = count_params(params)
    print("="*60)
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print("="*60 + "\n")


class MovingAverage:
    """Exponential moving average for metrics."""
    
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.value = None
    
    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * new_value
        return self.value
    
    def get(self) -> float:
        return self.value if self.value is not None else 0.0


class MetricsTracker:
    """Track multiple metrics with moving averages."""
    
    def __init__(self, decay: float = 0.99):
        self.metrics = {}
        self.decay = decay
    
    def update(self, metrics_dict: dict):
        """Update all metrics."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = MovingAverage(self.decay)
            self.metrics[key].update(float(value))
    
    def get_all(self) -> dict:
        """Get all current metric values."""
        return {key: ma.get() for key, ma in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
