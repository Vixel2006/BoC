from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class VQConfig:
    """Vector Quantization configuration."""
    num_embeddings: int = 512  # Codebook size
    commitment_cost: float = 0.25  # Commitment loss weight
    decay: float = 0.99  # EMA decay
    epsilon: float = 1e-5
    use_entropy_loss: bool = True
    entropy_loss_weight: float = 0.1
    code_reset_threshold: float = 0.01  # Reset codes used less than this


@dataclass
class ImageEncoderConfig:
    """ViT image encoder configuration."""
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 6
    mlp_dim: int = 1536
    dropout_rate: float = 0.1


@dataclass
class ImageDecoderConfig:
    """VAE image decoder configuration."""
    use_simple_vae: bool = True
    base_channels: int = 64
    num_upsample: int = 4  # Number of 2x upsample layers
    hidden_dims: Tuple[int, ...] = (256, 128, 64, 32)  # For full VAE


@dataclass
class TextEncoderConfig:
    """Text encoder configuration."""
    vocab_size: int = 30000
    max_len: int = 128
    embed_dim: int = 384  # Should match image encoder
    num_heads: int = 6
    num_layers: int = 6
    mlp_dim: int = 1536
    dropout_rate: float = 0.1


@dataclass
class TextDecoderConfig:
    """Text decoder configuration."""
    vocab_size: int = 30000
    max_len: int = 128
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 4  # Typically fewer layers than encoder
    mlp_dim: int = 1536
    dropout_rate: float = 0.1


@dataclass
class LossConfig:
    """Loss function weights and parameters."""
    # Image autoencoder
    image_reconstruction_weight: float = 1.0
    image_reconstruction_type: str = 'mse'  # 'mse' or 'l1'
    
    # Text autoencoder
    text_reconstruction_weight: float = 1.0
    pad_token_id: int = 0
    
    # VQ losses
    vq_weight: float = 1.0
    commitment_weight: float = 0.25
    entropy_weight: float = 0.1
    
    # Cross-modal alignment
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0
    symmetric_contrastive: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training phases
    phase_1_steps: int = 50000  # Image autoencoder only
    phase_2_steps: int = 50000  # Text encoder + decoder with alignment
    phase_3_steps: int = 20000  # Joint fine-tuning (optional)
    
    # Optimization
    learning_rate: float = 3e-4
    warmup_steps: int = 5000
    weight_decay: float = 0.01
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Batch sizes
    batch_size_phase1: int = 64  # Image autoencoder
    batch_size_phase2: int = 32  # Text + alignment
    batch_size_phase3: int = 32  # Joint
    
    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 5000
    checkpoint_dir: str = './checkpoints'
    
    # Data
    train_data_path: str = './data/train'
    eval_data_path: str = './data/eval'
    num_workers: int = 4
    
    # Device
    use_mixed_precision: bool = False
    seed: int = 42


@dataclass
class BoCConfig:
    """Complete BoC model configuration."""
    vq: VQConfig = field(default_factory=VQConfig)
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    image_decoder: ImageDecoderConfig = field(default_factory=ImageDecoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    text_decoder: TextDecoderConfig = field(default_factory=TextDecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Validate configuration consistency."""
        # Ensure embed_dim matches across components
        assert self.image_encoder.embed_dim == self.text_encoder.embed_dim, \
            "Image and text encoder embed_dim must match"
        assert self.image_encoder.embed_dim == self.text_decoder.embed_dim, \
            "Encoder and decoder embed_dim must match"
        
        # Ensure vocab_size matches
        assert self.text_encoder.vocab_size == self.text_decoder.vocab_size, \
            "Text encoder and decoder vocab_size must match"
        
        # Ensure max_len matches
        assert self.text_encoder.max_len == self.text_decoder.max_len, \
            "Text encoder and decoder max_len must match"
    
    def to_model_kwargs(self) -> dict:
        """Convert config to BoCModel initialization kwargs."""
        return {
            # Image config
            'image_size': self.image_encoder.image_size,
            'patch_size': self.image_encoder.patch_size,
            'image_channels': self.image_encoder.channels,
            
            # Text config
            'vocab_size': self.text_encoder.vocab_size,
            'max_text_len': self.text_encoder.max_len,
            
            # Shared embedding
            'num_embeddings': self.vq.num_embeddings,
            'embed_dim': self.image_encoder.embed_dim,
            
            # Encoder architecture
            'num_heads': self.image_encoder.num_heads,
            'num_layers': self.image_encoder.num_layers,
            'mlp_dim': self.image_encoder.mlp_dim,
            
            # Decoder architecture
            'decoder_num_layers': self.text_decoder.num_layers,
            
            # VAE decoder
            'use_simple_vae': self.image_decoder.use_simple_vae,
            'vae_base_channels': self.image_decoder.base_channels,
            
            # VQ parameters
            'vq_commitment_cost': self.vq.commitment_cost,
            'vq_decay': self.vq.decay,
            
            # Other
            'dropout_rate': self.image_encoder.dropout_rate,
        }


# Preset configurations for different model sizes

def get_small_config() -> BoCConfig:
    """Small model for fast experimentation."""
    config = BoCConfig()
    config.image_encoder.embed_dim = 256
    config.image_encoder.num_heads = 4
    config.image_encoder.num_layers = 4
    config.image_encoder.mlp_dim = 1024
    
    config.text_encoder.embed_dim = 256
    config.text_encoder.num_heads = 4
    config.text_encoder.num_layers = 4
    config.text_encoder.mlp_dim = 1024
    
    config.text_decoder.embed_dim = 256
    config.text_decoder.num_heads = 4
    config.text_decoder.num_layers = 3
    config.text_decoder.mlp_dim = 1024
    
    config.vq.num_embeddings = 256
    
    return config


def get_base_config() -> BoCConfig:
    """Base model (default)."""
    return BoCConfig()


def get_large_config() -> BoCConfig:
    """Large model for better performance."""
    config = BoCConfig()
    config.image_encoder.embed_dim = 768
    config.image_encoder.num_heads = 12
    config.image_encoder.num_layers = 12
    config.image_encoder.mlp_dim = 3072
    
    config.text_encoder.embed_dim = 768
    config.text_encoder.num_heads = 12
    config.text_encoder.num_layers = 12
    config.text_encoder.mlp_dim = 3072
    
    config.text_decoder.embed_dim = 768
    config.text_decoder.num_heads = 12
    config.text_decoder.num_layers = 6
    config.text_decoder.mlp_dim = 3072
    
    config.vq.num_embeddings = 1024
    config.image_decoder.base_channels = 128
    
    return config
