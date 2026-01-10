"""
Training script for Bag of Concepts (BoC) model.

Multi-phase training:
1. Phase 1: Image autoencoder (ViT → VQ → VAE)
2. Phase 2: Text encoder + decoder with cross-modal alignment
3. Phase 3: Joint fine-tuning (optional)
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import Dict, Any, Tuple
from pathlib import Path
import time

from ..models import BoCModel
from .config import BoCConfig, get_base_config
from .losses import (
    info_nce_loss,
    reconstruction_loss_image,
    reconstruction_loss_text,
    compute_codebook_metrics,
)


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""
    step_phase: int = 0  # Which training phase (1, 2, or 3)
    

def create_learning_rate_schedule(
    base_lr: float,
    warmup_steps: int,
    total_steps: int
) -> optax.Schedule:
    """Create learning rate schedule with linear warmup and cosine decay."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=warmup_steps
    )
    
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )


def create_train_state(
    rng: jax.Array,
    config: BoCConfig,
    phase: int = 1
) -> TrainState:
    """
    Create training state for a specific phase.
    
    Args:
        rng: Random number generator
        config: Model configuration
        phase: Training phase (1, 2, or 3)
    """
    # Initialize model
    model = BoCModel(**config.to_model_kwargs())
    
    # Create dummy inputs for initialization
    img_rng, text_rng = jax.random.split(rng)
    dummy_images = jax.random.normal(
        img_rng,
        (1, config.image_encoder.image_size, config.image_encoder.image_size, 3)
    )
    dummy_text = jax.random.randint(
        text_rng,
        (1, config.text_encoder.max_len),
        0,
        config.text_encoder.vocab_size
    )
    
    # Initialize parameters
    variables = model.init(
        rng,
        images=dummy_images,
        text_tokens=dummy_text,
        training=False,
        mode='both'
    )
    
    # Create optimizer
    total_steps = {
        1: config.training.phase_1_steps,
        2: config.training.phase_2_steps,
        3: config.training.phase_3_steps,
    }[phase]
    
    lr_schedule = create_learning_rate_schedule(
        base_lr=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.training.adam_b1,
            b2=config.training.adam_b2,
            eps=config.training.adam_eps,
            weight_decay=config.training.weight_decay
        )
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        step_phase=phase
    )


# Phase 1: Image Autoencoder Training
def train_step_phase1(
    state: TrainState,
    batch: Dict[str, jax.Array],
    config: BoCConfig
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Training step for Phase 1: Image autoencoder.
    
    Args:
        state: Current training state
        batch: Batch with 'images' key
        config: Model configuration
    """
    def loss_fn(params):
        # Forward pass
        outputs = state.apply_fn(
            {'params': params},
            images=batch['images'],
            training=True,
            mode='image_ae'
        )
        
        reconstructed = outputs['reconstructed_images']
        vq_info = outputs['image_vq_info']
        
        # Reconstruction loss
        recon_loss = reconstruction_loss_image(
            reconstructed,
            batch['images'],
            loss_type=config.loss.image_reconstruction_type
        )
        
        # VQ loss
        vq_loss = vq_info['vq_loss']
        
        # Total loss
        total_loss = (
            config.loss.image_reconstruction_weight * recon_loss +
            config.loss.vq_weight * vq_loss
        )
        
        # Metrics
        metrics = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'commitment_loss': vq_info['commitment_loss'],
            'perplexity': vq_info['perplexity'],
        }
        
        # Codebook metrics
        codebook_metrics = compute_codebook_metrics(
            vq_info['avg_probs'],
            config.vq.num_embeddings
        )
        metrics.update(codebook_metrics)
        
        return total_loss, metrics
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


# Phase 2: Text Encoder + Decoder with Alignment
def train_step_phase2(
    state: TrainState,
    batch: Dict[str, jax.Array],
    config: BoCConfig
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Training step for Phase 2: Text autoencoder with alignment.
    
    Args:
        state: Current training state
        batch: Batch with 'images' and 'text_tokens' keys
        config: Model configuration
    """
    def loss_fn(params):
        # Text autoencoder
        text_outputs = state.apply_fn(
            {'params': params},
            text_tokens=batch['text_tokens'],
            training=True,
            mode='text_ae'
        )
        
        text_logits = text_outputs['text_logits']
        text_vq_info = text_outputs['text_vq_info']
        
        # Text reconstruction loss
        text_recon_loss, text_metrics = reconstruction_loss_text(
            text_logits,
            batch['text_tokens'],
            pad_token_id=config.loss.pad_token_id
        )
        
        # Cross-modal alignment
        alignment_outputs = state.apply_fn(
            {'params': params},
            images=batch['images'],
            text_tokens=batch['text_tokens'],
            training=True,
            mode='alignment'
        )
        
        image_emb = alignment_outputs['image_embeddings']
        text_emb = alignment_outputs['text_embeddings']
        
        # InfoNCE loss
        contrastive_losses = info_nce_loss(
            image_emb,
            text_emb,
            temperature=config.loss.contrastive_temperature,
            symmetric=config.loss.symmetric_contrastive
        )
        
        # Total loss
        total_loss = (
            config.loss.text_reconstruction_weight * text_recon_loss +
            config.loss.vq_weight * text_vq_info['vq_loss'] +
            config.loss.contrastive_weight * contrastive_losses['loss']
        )
        
        # Metrics
        metrics = {
            'total_loss': total_loss,
            'text_recon_loss': text_recon_loss,
            'text_perplexity': text_metrics['perplexity'],
            'text_accuracy': text_metrics['accuracy'],
            'text_vq_loss': text_vq_info['vq_loss'],
            'contrastive_loss': contrastive_losses['loss'],
            'i2t_loss': contrastive_losses['image_to_text_loss'],
            't2i_loss': contrastive_losses['text_to_image_loss'],
        }
        
        return total_loss, metrics
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


# Phase 3: Joint fine-tuning
def train_step_phase3(
    state: TrainState,
    batch: Dict[str, jax.Array],
    config: BoCConfig
) -> Tuple[TrainState, Dict[str, float]]:
    """
    Training step for Phase 3: Joint fine-tuning.
    
    This combines all objectives from phases 1 and 2.
    """
    def loss_fn(params):
        # Image autoencoder
        img_outputs = state.apply_fn(
            {'params': params},
            images=batch['images'],
            training=True,
            mode='image_ae'
        )
        
        img_recon = img_outputs['reconstructed_images']
        img_vq_info = img_outputs['image_vq_info']
        
        img_recon_loss = reconstruction_loss_image(
            img_recon,
            batch['images'],
            loss_type=config.loss.image_reconstruction_type
        )
        
        # Text autoencoder
        text_outputs = state.apply_fn(
            {'params': params},
            text_tokens=batch['text_tokens'],
            training=True,
            mode='text_ae'
        )
        
        text_logits = text_outputs['text_logits']
        text_vq_info = text_outputs['text_vq_info']
        
        text_recon_loss, text_metrics = reconstruction_loss_text(
            text_logits,
            batch['text_tokens'],
            pad_token_id=config.loss.pad_token_id
        )
        
        # Cross-modal alignment
        alignment_outputs = state.apply_fn(
            {'params': params},
            images=batch['images'],
            text_tokens=batch['text_tokens'],
            training=True,
            mode='alignment'
        )
        
        contrastive_losses = info_nce_loss(
            alignment_outputs['image_embeddings'],
            alignment_outputs['text_embeddings'],
            temperature=config.loss.contrastive_temperature,
            symmetric=config.loss.symmetric_contrastive
        )
        
        # Total loss
        total_loss = (
            config.loss.image_reconstruction_weight * img_recon_loss +
            config.loss.text_reconstruction_weight * text_recon_loss +
            config.loss.vq_weight * (img_vq_info['vq_loss'] + text_vq_info['vq_loss']) / 2 +
            config.loss.contrastive_weight * contrastive_losses['loss']
        )
        
        metrics = {
            'total_loss': total_loss,
            'img_recon_loss': img_recon_loss,
            'text_recon_loss': text_recon_loss,
            'contrastive_loss': contrastive_losses['loss'],
            'img_perplexity': img_vq_info['perplexity'],
            'text_perplexity': text_vq_info['perplexity'],
        }
        
        return total_loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


def train_phase(
    state: TrainState,
    train_dataloader,
    config: BoCConfig,
    phase: int
):
    """
    Train a specific phase.
    
    Args:
        state: Initial training state
        train_dataloader: Data loader
        config: Configuration
        phase: Phase number (1, 2, or 3)
    """
    # Select appropriate training step function
    train_step = {
        1: train_step_phase1,
        2: train_step_phase2,
        3: train_step_phase3,
    }[phase]
    
    num_steps = {
        1: config.training.phase_1_steps,
        2: config.training.phase_2_steps,
        3: config.training.phase_3_steps,
    }[phase]
    
    print(f"\n{'='*60}")
    print(f"Starting Phase {phase} Training")
    print(f"{'='*60}\n")
    
    for step in range(num_steps):
        # Get batch
        batch = next(train_dataloader)
        
        # Training step
        state, metrics = train_step(state, batch, config)
        
        # Logging
        if step % config.training.log_every == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"Phase {phase} | Step {step}/{num_steps} | {metrics_str}")
        
        # Checkpointing
        if step % config.training.save_every == 0 and step > 0:
            save_checkpoint(state, config, phase, step)
    
    return state


def save_checkpoint(state: TrainState, config: BoCConfig, phase: int, step: int):
    """Save checkpoint."""
    ckpt_dir = Path(config.training.checkpoint_dir) / f"phase_{phase}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints.save_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target=state,
        step=step,
        overwrite=True,
        keep=3
    )
    print(f"Checkpoint saved at step {step}")


def main():
    """Main training loop."""
    # Configuration
    config = get_base_config()
    
    # Random seed
    rng = jax.random.PRNGKey(config.training.seed)
    
    # TODO: Initialize dataloaders
    # train_dataloader = create_dataloader(config)
    
    print("BoC Training Start")
    print(f"Model: {config.vq.num_embeddings} codes, {config.image_encoder.embed_dim}D")
    
    # Phase 1: Image Autoencoder
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config, phase=1)
    # state = train_phase(state, train_dataloader, config, phase=1)
    
    # Phase 2: Text + Alignment
    state.step_phase = 2
    # state = train_phase(state, train_dataloader, config, phase=2)
    
    # Phase 3: Joint fine-tuning (optional)
    # state.step_phase = 3
    # state = train_phase(state, train_dataloader, config, phase=3)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
