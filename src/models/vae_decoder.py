import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

class ResidualBlock(nn.Module):
    """Residual block for VAE decoder."""
    features: int
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.gelu(x)
        
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.GroupNorm(num_groups=8)(x)
        
        # Match dimensions if needed
        if residual.shape[-1] != self.features:
            residual = nn.Conv(features=self.features, kernel_size=(1, 1))(residual)
        
        return nn.gelu(x + residual)


class VAEDecoder(nn.Module):
    """
    VAE Decoder for reconstructing images from VQ codes.
    
    Architecture:
    - Takes VQ embeddings of shape (B, H', W', embed_dim)
    - Upsamples through transpose convolutions
    - Uses residual blocks for stable training
    - Outputs images of shape (B, H, W, C)
    
    Args:
        embed_dim: Dimension of VQ embeddings
        hidden_dims: List of hidden dimensions for upsampling layers
        output_channels: Number of output channels (3 for RGB)
        image_size: Target image size (assumes square images)
        patch_size: Patch size used in ViT encoder (for calculating initial spatial size)
    """
    embed_dim: int
    hidden_dims: Tuple[int, ...] = (256, 128, 64, 32)
    output_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    
    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """
        Args:
            x: VQ embeddings of shape (B, H', W', embed_dim) where:
               H' = W' = image_size // patch_size
            training: Whether in training mode
            
        Returns:
            Reconstructed image of shape (B, image_size, image_size, output_channels)
        """
        # Initial projection to first hidden dimension
        x = nn.Conv(
            features=self.hidden_dims[0],
            kernel_size=(3, 3),
            padding='SAME',
            name='initial_conv'
        )(x)
        x = nn.GroupNorm(num_groups=8)(x)
        x = nn.gelu(x)
        
        # Residual block at current resolution
        x = ResidualBlock(features=self.hidden_dims[0])(x)
        
        # Upsampling layers
        for i, hidden_dim in enumerate(self.hidden_dims[1:], 1):
            # Upsample 2x using transpose convolution
            x = nn.ConvTranspose(
                features=hidden_dim,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                name=f'upsample_{i}'
            )(x)
            x = nn.GroupNorm(num_groups=8)(x)
            x = nn.gelu(x)
            
            # Residual block
            x = ResidualBlock(features=hidden_dim)(x)
        
        # Final convolution to RGB
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            padding='SAME',
            name='output_conv'
        )(x)
        
        # Tanh activation to constrain output to [-1, 1]
        x = nn.tanh(x)
        
        return x


class SimpleVAEDecoder(nn.Module):
    """
    Simpler VAE Decoder with fewer parameters for faster experimentation.
    
    Args:
        embed_dim: Dimension of VQ embeddings
        base_channels: Base number of channels (will be scaled up)
        output_channels: Number of output channels (3 for RGB)
        num_upsample: Number of upsampling layers
    """
    embed_dim: int
    base_channels: int = 64
    output_channels: int = 3
    num_upsample: int = 4  # 4 upsample layers: 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    
    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        """
        Args:
            x: VQ embeddings of shape (B, H', W', embed_dim)
            training: Whether in training mode
            
        Returns:
            Reconstructed image
        """
        # Initial projection
        x = nn.Conv(
            features=self.base_channels * (2 ** (self.num_upsample - 1)),
            kernel_size=(3, 3),
            padding='SAME'
        )(x)
        x = nn.gelu(x)
        
        # Progressive upsampling
        for i in range(self.num_upsample):
            scale = 2 ** (self.num_upsample - 1 - i)
            features = self.base_channels * scale
            
            x = nn.ConvTranspose(
                features=features,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME'
            )(x)
            x = nn.GroupNorm(num_groups=min(8, features))(x)
            x = nn.gelu(x)
        
        # Output layer
        x = nn.Conv(
            features=self.output_channels,
            kernel_size=(3, 3),
            padding='SAME'
        )(x)
        x = nn.tanh(x)
        
        return x
