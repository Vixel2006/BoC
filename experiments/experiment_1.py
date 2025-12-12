import torch

from src.train import main as train_main
from configs.training_config import TrainingConfig
from configs.bag_of_concepts_config import BagOfConceptsConfig
from configs.concept_mapper_config import ConceptMapperConfig
from configs.image_decoder_config import ImageDecoderConfig
from configs.text_decoder_config import TextDecoderConfig

def main():
    """
    Defines the configuration for the first experiment and runs the training.
    """
    print("Starting experiment_1.py: Defining configurations and running training.")

    # 1. Define config classes with our experiment hyperparameters
    # Bag of Concepts Configuration
    boc_config = BagOfConceptsConfig(
        num_concepts=256,  # Example: Reduced number of concepts
        concept_dim=512,
        commitment=0.5,    # Example: Adjusted commitment
    )

    # Concept Mapper Configurations (Image and Text)
    # Assuming default values for num_attention_heads and multi_head_attention_dropout_rate
    img_mapper_config = ConceptMapperConfig(
        modality="image",
        # You can specify image_encoder_config here if needed, e.g.,
        # image_encoder_config=ImageEncoderConfig(pretrained_model_name_or_path="openai/clip-vit-base-patch32")
    )
    txt_mapper_config = ConceptMapperConfig(
        modality="text",
        # You can specify text_encoder_config here if needed, e.g.,
        # text_encoder_config=TextEncoderConfig(pretrained_model_name_or_path="bert-base-uncased")
    )

    # Image Decoder Configuration
    img_decoder_config = ImageDecoderConfig(
        image_size=64,
        num_decoder_layers=4,
        input_dim=512,
        hidden_dim=512,
    )

    # Text Decoder Configuration
    # vocab_size is crucial and should match the tokenizer used in src/train.py (bert-base-uncased)
    txt_decoder_config = TextDecoderConfig(
        vocab_size=30522, # Vocab size for 'bert-base-uncased'
        hidden_dim=512,
        num_decoder_layers=1,
        input_dim=512,
        max_seq_len=20,
    )

    # Main Training Configuration
    training_config = TrainingConfig(
        epochs=5,  # Run for a shorter duration for this first experiment
        device="cuda" if torch.cuda.is_available() else "cpu",
        bag_of_concepts=boc_config,
        concept_mapper_image=img_mapper_config,
        concept_mapper_text=txt_mapper_config,
        image_decoder=img_decoder_config,
        text_decoder=txt_decoder_config,
    )

    print("\nExperiment Configuration:")
    print(training_config)

    # 2. Run the training loop with those configs and run the evaluation
    print("\nStarting training...")
    metrics = train_main(training_config)
    print("\nTraining finished.")

    # 3. Display evaluation results
    print("\n--- Evaluation Results ---")
    if metrics:
        for metric_name, values in metrics.items():
            if values: # Check if the list of values is not empty
                print(f"{metric_name}: {values[-1]:.4f} (last epoch)")
            else:
                print(f"{metric_name}: No data collected")
    else:
        print("No metrics returned from training.")

if __name__ == "__main__":
    main()
