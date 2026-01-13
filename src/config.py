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
class ViTConfig:
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
class TextFormerConfig:
    """Text encoder configuration."""
    vocab_size: int = 30000
    max_len: int = 128
    embed_dim: int = 384  # Should match image encoder
    num_heads: int = 6
    num_layers: int = 6
    mlp_dim: int = 1536
    dropout_rate: float = 0.1


@dataclass
class LossConfig:
    """Loss function weights and parameters."""
    # VQ losses
    vq_weight: float = 1.0
    commitment_weight: float = 0.25
    entropy_weight: float = 0.1
    
    # Cross-modal alignment
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0
    symmetric_contrastive: bool = True
    
    # Removed reconstruction weights


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training phases (Modified semantics)
    phase_1_steps: int = 1000  # Image encoder VQ only (fast)
    phase_2_steps: int = 50000  # Text encoder + Alignment
    phase_3_steps: int = 20000  # Joint fine-tuning
    
    # Optimization
    learning_rate: float = 3e-4
    warmup_steps: int = 5000
    weight_decay: float = 0.01
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    
    # Batch sizes
    batch_size_phase1: int = 64
    batch_size_phase2: int = 32
    batch_size_phase3: int = 32
    
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
    
    config.vq.num_embeddings = 1024
    
    return config
