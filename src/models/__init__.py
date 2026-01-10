"""BoC Models Package - Neural network components for Bag of Concepts."""

from .vq import VectorQuantizer, update_codebook_ema
from .transformer import MLP, EncoderBlock, DecoderBlock
from .vit_encoder import ViTEncoder, ViT
from .text_encoder import TextEncoder, TextTransformer
from .vae_decoder import VAEDecoder, SimpleVAEDecoder, ResidualBlock
from .text_decoder import TextDecoder
from .boc_model import BoCModel

__all__ = [
    # VQ
    'VectorQuantizer',
    'update_codebook_ema',
    # Transformer blocks
    'MLP',
    'EncoderBlock',
    'DecoderBlock',
    # Encoders
    'ViTEncoder',
    'ViT',
    'TextEncoder',
    'TextTransformer',
    # Decoders
    'VAEDecoder',
    'SimpleVAEDecoder',
    'ResidualBlock',
    'TextDecoder',
    # Main model
    'BoCModel',
]
