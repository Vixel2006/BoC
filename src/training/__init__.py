"""BoC Training Package - Training loops, losses, and configurations."""

from .losses import (
    info_nce_loss,
    reconstruction_loss_image,
    reconstruction_loss_text,
    compute_codebook_metrics,
    total_vq_autoencoder_loss,
)
from .config import (
    VQConfig,
    ImageEncoderConfig,
    ImageDecoderConfig,
    TextEncoderConfig,
    TextDecoderConfig,
    LossConfig,
    TrainingConfig,
    BoCConfig,
    get_small_config,
    get_base_config,
    get_large_config,
)
from .train import (
    TrainState,
    create_train_state,
    train_step_phase1,
    train_step_phase2,
    train_step_phase3,
    train_phase,
)

__all__ = [
    # Losses
    'info_nce_loss',
    'reconstruction_loss_image',
    'reconstruction_loss_text',
    'compute_codebook_metrics',
    'total_vq_autoencoder_loss',
    # Configs
    'VQConfig',
    'ImageEncoderConfig',
    'ImageDecoderConfig',
    'TextEncoderConfig',
    'TextDecoderConfig',
    'LossConfig',
    'TrainingConfig',
    'BoCConfig',
    'get_small_config',
    'get_base_config',
    'get_large_config',
    # Training
    'TrainState',
    'create_train_state',
    'train_step_phase1',
    'train_step_phase2',
    'train_step_phase3',
    'train_phase',
]
