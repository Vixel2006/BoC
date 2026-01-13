import jax
import jax.numpy as jnp
from src.models import BoCModel
from src.config import get_base_config

def main():
    print("="*60)
    print("Bag of Concepts (BoC) - Minimal Mode")
    print("="*60)
    
    # Configuration
    config = get_base_config()
    
    print(f"\nModel Configuration:")
    print(f"- Image Embed Dim: {config.image_encoder.embed_dim}")
    print(f"- VQ Codebook Size: {config.vq.num_embeddings}")
    
    # Initialize model
    print("\nInitializing model...")
    rng = jax.random.PRNGKey(config.training.seed)
    model = BoCModel(**config.to_model_kwargs())
    
    # Create dummy inputs
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
    
    print("✓ Model initialized successfully.")
    
    # Forward pass check
    print("\nRunning forward pass...")
    outputs = model.apply(
        variables,
        images=dummy_images,
        text_tokens=dummy_text,
        training=False,
        mode='both'
    )
    
    print("✓ Forward pass complete.")
    print("Outputs:", list(outputs.keys()))


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
