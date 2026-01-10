"""
Bag of Concepts (BoC) - Main training and inference script.

This script provides a command-line interface for:
- Training the model on Flickr30k or MS COCO
- Evaluating trained models
- Running inference (text→image, image→text)

Examples:
    # Train on Flickr30k
    python main.py train --dataset flickr30k --config base --data-root ./data/flickr30k
    
    # Evaluate on COCO
    python main.py eval --dataset coco --checkpoint ./checkpoints/phase_2/ckpt_50000
    
    # Generate image from text
    python main.py generate-image --checkpoint ./checkpoints/best --text "A dog playing in the park"
    
    # Generate caption from image
    python main.py generate-text --checkpoint ./checkpoints/best --image ./test.jpg
"""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import json
import pickle
from datetime import datetime

from src.models import BoCModel
from src.training import (
    get_small_config,
    get_base_config,
    get_large_config,
    create_train_state,
    train_phase,
)
from src.data import (
    DataConfig,
    create_dataloader,
    build_tokenizer_from_dataset,
    SimpleTokenizer,
    verify_dataset,
)
from src.utils import (
    load_image,
    save_image,
    visualize_reconstruction,
    visualize_codebook_usage,
    MetricsTracker,
)


def setup_experiment_dir(output_dir: str, experiment_name: str) -> Path:
    """Create experiment directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    return exp_dir


def train_command(args):
    """Training command."""
    print("="*60)
    print("BoC Training")
    print("="*60)
    
    # Verify dataset exists
    print(f"\nChecking dataset: {args.dataset}")
    if not verify_dataset(args.dataset, args.data_root):
        print(f"\n❌ Dataset not found at: {args.data_root}")
        print("\nTo download the dataset, run:")
        print(f"  python scripts/download_datasets.py --dataset {args.dataset} --output-dir {args.data_root}")
        print("\nOr manually prepare the dataset following README.md instructions.")
        return
    print(f"✓ Dataset verified: {args.data_root}")
    
    # Load config
    if args.config == 'small':
        config = get_small_config()
    elif args.config == 'base':
        config = get_base_config()
    elif args.config == 'large':
        config = get_large_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Update config with CLI args
    if args.batch_size:
        config.training.batch_size_phase1 = args.batch_size
        config.training.batch_size_phase2 = args.batch_size
        config.training.batch_size_phase3 = args.batch_size
    
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(args.output_dir, args.experiment_name or args.dataset)
    config.training.checkpoint_dir = str(exp_dir / "checkpoints")
    
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Model: {config.image_encoder.embed_dim}D, {config.vq.num_embeddings} codes")
    print(f"Dataset: {args.dataset}")
    
    # Save config
    with open(exp_dir / "config.json", 'w') as f:
        # Convert dataclasses to dict for JSON serialization
        config_dict = {
            'model_size': args.config,
            'dataset': args.dataset,
            'vq': config.vq.__dict__,
            'training': config.training.__dict__,
        }
        json.dump(config_dict, f, indent=2)
    
    # Build or load tokenizer
    tokenizer_path = exp_dir / "tokenizer.pkl"
    if tokenizer_path.exists() and not args.rebuild_vocab:
        print("\nLoading existing tokenizer...")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        print("\nBuilding tokenizer from dataset...")
        data_config = DataConfig(
            dataset_name=args.dataset,
            data_root=args.data_root,
            split='train',
            vocab_size=config.text_encoder.vocab_size,
        )
        tokenizer = build_tokenizer_from_dataset(data_config)
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_path}")
    
    # Create dataloaders
    print("\n Creating dataloaders...")
    train_config = DataConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split='train',
        image_size=config.image_encoder.image_size,
        max_text_len=config.text_encoder.max_len,
        batch_size=config.training.batch_size_phase1,
        shuffle=True,
    )
    
    val_config = DataConfig(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split='val',
        image_size=config.image_encoder.image_size,
        max_text_len=config.text_encoder.max_len,
        batch_size=config.training.batch_size_phase1,
        shuffle=False,
    )
    
    # Initialize model
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(config.training.seed)
    
    # Start training based on phase
    if args.phase == 1 or args.phase == 'all':
        print("\n" + "="*60)
        print("Phase 1: Image Autoencoder Training")
        print("="*60)
        
        state = create_train_state(rng, config, phase=1)
        train_dataloader = create_dataloader(train_config, tokenizer)
        state = train_phase(state, train_dataloader, config, phase=1)
    
    if args.phase == 2 or args.phase == 'all':
        print("\n" + "="*60)
        print("Phase 2: Text Encoder + Alignment")
        print("="*60)
        
        if args.phase == 2 and args.resume_checkpoint:
            # Load checkpoint from phase 1
            # TODO: Implement checkpoint loading
            pass
        
        train_config.batch_size = config.training.batch_size_phase2
        train_dataloader = create_dataloader(train_config, tokenizer)
        
        if 'state' not in locals():
            state = create_train_state(rng, config, phase=2)
        else:
            state.step_phase = 2
        
        state = train_phase(state, train_dataloader, config, phase=2)
    
    if args.phase == 3 or args.phase == 'all':
        print("\n" + "="*60)
        print("Phase 3: Joint Fine-tuning")
        print("="*60)
        
        train_config.batch_size = config.training.batch_size_phase3
        train_dataloader = create_dataloader(train_config, tokenizer)
        
        if 'state' not in locals():
            state = create_train_state(rng, config, phase=3)
        else:
            state.step_phase = 3
        
        state = train_phase(state, train_dataloader, config, phase=3)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")
    print("="*60)


def eval_command(args):
    """Evaluation command."""
    print("="*60)
    print("BoC Evaluation")
    print("="*60)
    
    # TODO: Implement evaluation
    # - Load checkpoint
    # - Load test dataset
    # - Compute metrics:
    #   - Image reconstruction (PSNR, SSIM)
    #   - Text reconstruction (perplexity, BLEU)
    #   - Retrieval (recall@k)
    #   - FID for generated images
    
    print("\nEvaluation not yet implemented.")
    print("TODO: Add metrics computation")


def generate_image_command(args):
    """Generate image from text."""
    print("="*60)
    print("Text → Image Generation")
    print("="*60)
    
    print(f"\nInput: {args.text}")
    
    # TODO: Implement text-to-image generation
    # - Load checkpoint
    # - Load tokenizer
    # - Tokenize text
    # - Generate image
    # - Save result
    
    print("\nGeneration not yet implemented.")
    print("TODO: Implement inference pipeline")


def generate_text_command(args):
    """Generate text from image."""
    print("="*60)
    print("Image → Text Generation")
    print("="*60)
    
    print(f"\nInput image: {args.image}")
    
    # TODO: Implement image-to-text generation
    # - Load checkpoint
    # - Load tokenizer
    # - Load and preprocess image
    # - Generate text
    # - Print result
    
    print("\nGeneration not yet implemented.")
    print("TODO: Implement inference pipeline")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bag of Concepts - Multimodal Concept-based Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--dataset', type=str, required=True, choices=['flickr30k', 'coco'],
                             help='Dataset to use')
    train_parser.add_argument('--data-root', type=str, required=True,
                             help='Path to dataset root directory')
    train_parser.add_argument('--config', type=str, default='base', choices=['small', 'base', 'large'],
                             help='Model configuration')
    train_parser.add_argument('--phase', type=str, default='all', choices=['1', '2', '3', 'all'],
                             help='Training phase (1: image AE, 2: text+align, 3: joint, all: all phases)')
    train_parser.add_argument('--output-dir', type=str, default='./experiments',
                             help='Output directory for checkpoints and logs')
    train_parser.add_argument('--experiment-name', type=str, default=None,
                             help='Experiment name (default: dataset name)')
    train_parser.add_argument('--batch-size', type=int, default=None,
                             help='Batch size (overrides config)')
    train_parser.add_argument('--learning-rate', type=float, default=None,
                             help='Learning rate (overrides config)')
    train_parser.add_argument('--resume-checkpoint', type=str, default=None,
                             help='Resume from checkpoint')
    train_parser.add_argument('--rebuild-vocab', action='store_true',
                             help='Rebuild vocabulary even if tokenizer exists')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--dataset', type=str, required=True, choices=['flickr30k', 'coco'])
    eval_parser.add_argument('--data-root', type=str, required=True)
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to checkpoint')
    eval_parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    eval_parser.add_argument('--output-file', type=str, default='results.json')
    
    # Generate image command
    gen_img_parser = subparsers.add_parser('generate-image', help='Generate image from text')
    gen_img_parser.add_argument('--checkpoint', type=str, required=True)
    gen_img_parser.add_argument('--text', type=str, required=True,
                               help='Text description')
    gen_img_parser.add_argument('--output', type=str, default='generated_image.png')
    gen_img_parser.add_argument('--temperature', type=float, default=1.0)
    
    # Generate text command
    gen_text_parser = subparsers.add_parser('generate-text', help='Generate text from image')
    gen_text_parser.add_argument('--checkpoint', type=str, required=True)
    gen_text_parser.add_argument('--image', type=str, required=True,
                                help='Path to input image')
    gen_text_parser.add_argument('--max-length', type=int, default=128)
    gen_text_parser.add_argument('--temperature', type=float, default=0.9)
    gen_text_parser.add_argument('--top-k', type=int, default=50)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'generate-image':
        generate_image_command(args)
    elif args.command == 'generate-text':
        generate_text_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
