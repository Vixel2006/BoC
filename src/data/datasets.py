import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import json
from PIL import Image
from dataclasses import dataclass
import random


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    dataset_name: str  # 'flickr30k' or 'coco'
    data_root: str
    split: str  # 'train', 'val', or 'test'
    image_size: int = 224
    max_text_len: int = 128
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    vocab_size: int = 30000
    tokenizer_path: Optional[str] = None


class Flickr30kDataset:
    """
    Flickr30k dataset loader.
    
    Expected directory structure:
    data_root/
    ├── flickr30k_images/
    │   ├── 1000092795.jpg
    │   ├── 10002456.jpg
    │   └── ...
    └── flickr30k_annotations/
        ├── train.json
        ├── val.json
        └── test.json
    
    Annotation format:
    {
        "image_id": "1000092795.jpg",
        "captions": [
            "Two young guys with shaggy hair...",
            "Two young, White males are outside...",
            ...
        ]
    }
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_root = Path(config.data_root) / "flickr30k"
        self.image_dir = self.data_root / "flickr30k_images"
        self.annotation_file = self.data_root / "flickr30k_annotations" / f"{config.split}.json"
        
        # Load annotations
        with open(self.annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create image-caption pairs (each image has 5 captions)
        self.samples = []
        for ann in self.annotations:
            image_path = self.image_dir / ann['image_id']
            for caption in ann['captions']:
                self.samples.append({
                    'image_path': str(image_path),
                    'caption': caption,
                    'image_id': ann['image_id']
                })
        
        print(f"Loaded Flickr30k {config.split}: {len(self.samples)} image-caption pairs")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)
        image = np.array(image, dtype=np.float32) / 255.0
        image = image * 2.0 - 1.0  # Normalize to [-1, 1]
        
        return {
            'image': image,
            'caption': sample['caption'],
            'image_id': sample['image_id'],
        }


class COCODataset:
    """
    MS COCO dataset loader.
    
    Expected directory structure:
    data_root/
    ├── train2017/
    │   ├── 000000000009.jpg
    │   └── ...
    ├── val2017/
    │   └── ...
    └── annotations/
        ├── captions_train2017.json
        └── captions_val2017.json
    
    Uses official COCO annotation format.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_root = Path(config.data_root) / "coco"
        
        # Determine split directory
        if config.split == 'train':
            self.image_dir = self.data_root / 'train2017'
            ann_file = 'captions_train2017.json'
        elif config.split in ['val', 'test']:
            self.image_dir = self.data_root / 'val2017'
            ann_file = 'captions_val2017.json'
        else:
            raise ValueError(f"Unknown split: {config.split}")
        
        self.annotation_file = self.data_root / 'annotations' / ann_file
        
        # Load annotations
        with open(self.annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image_id to filename mapping
        self.id_to_filename = {
            img['id']: img['file_name']
            for img in coco_data['images']
        }
        
        # Create image-caption pairs
        self.samples = []
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in self.id_to_filename:
                self.samples.append({
                    'image_path': str(self.image_dir / self.id_to_filename[image_id]),
                    'caption': ann['caption'],
                    'image_id': image_id,
                })
        
        print(f"Loaded COCO {config.split}: {len(self.samples)} image-caption pairs")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        sample = self.samples[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)
        image = np.array(image, dtype=np.float32) / 255.0
        image = image * 2.0 - 1.0  # Normalize to [-1, 1]
        
        return {
            'image': image,
            'caption': sample['caption'],
            'image_id': sample['image_id'],
        }


class SimpleTokenizer:
    """
    Simple word-level tokenizer.
    
    For production, use a proper tokenizer like SentencePiece or HuggingFace.
    """
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<START>', 2: '<END>', 3: '<UNK>'}
        self.word_counts = {}
    
    def build_vocab(self, captions: List[str]):
        """Build vocabulary from captions."""
        # Count words
        for caption in captions:
            words = caption.lower().split()
            for word in words:
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Select most frequent words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.vocab_size - 4], start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"Built vocabulary: {len(self.word2idx)} words")
    
    def encode(self, caption: str, max_len: int = 128) -> np.ndarray:
        """Encode caption to token IDs."""
        words = caption.lower().split()
        tokens = [self.word2idx.get(word, 3) for word in words]  # 3 = <UNK>
        
        # Add START and END tokens
        tokens = [1] + tokens + [2]
        
        # Pad or truncate
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        return np.array(tokens, dtype=np.int32)
    
    def decode(self, tokens: np.ndarray) -> str:
        """Decode token IDs to caption."""
        words = []
        for token in tokens:
            if token == 0:  # PAD
                break
            if token == 1:  # START
                continue
            if token == 2:  # END
                break
            words.append(self.idx2word.get(int(token), '<UNK>'))
        return ' '.join(words)


def create_dataloader(
    config: DataConfig,
    tokenizer: SimpleTokenizer,
    seed: int = 42
) -> Iterator[Dict[str, jnp.ndarray]]:
    """
    Create dataloader iterator.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer for captions
        seed: Random seed
        
    Yields:
        Batches with 'images' and 'text_tokens'
    """
    # Create dataset
    if config.dataset_name == 'flickr30k':
        dataset = Flickr30kDataset(config)
    elif config.dataset_name == 'coco':
        dataset = COCODataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    # Create indices
    indices = list(range(len(dataset)))
    if config.shuffle:
        random.seed(seed)
        random.shuffle(indices)
    
    # Batch iteration
    batch_size = config.batch_size
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        # Load batch
        images = []
        captions = []
        
        for idx in batch_indices:
            sample = dataset[idx]
            images.append(sample['image'])
            captions.append(sample['caption'])
        
        # Tokenize captions
        text_tokens = np.stack([
            tokenizer.encode(caption, config.max_text_len)
            for caption in captions
        ])
        
        # Convert to JAX arrays
        images = jnp.array(np.stack(images))
        text_tokens = jnp.array(text_tokens)
        
        yield {
            'images': images,
            'text_tokens': text_tokens,
            'captions': captions,  # Keep for logging
        }


def build_tokenizer_from_dataset(config: DataConfig) -> SimpleTokenizer:
    """Build tokenizer from dataset captions."""
    # Load all captions
    if config.dataset_name == 'flickr30k':
        dataset = Flickr30kDataset(config)
    elif config.dataset_name == 'coco':
        dataset = COCODataset(config)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset_name}")
    
    captions = [sample['caption'] for sample in dataset.samples]
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(captions)
    
    return tokenizer


def verify_dataset(dataset_name: str, data_root: Path) -> bool:
    """Verify dataset is properly set up."""
    data_root = Path(data_root)  # Ensure Path object
    
    if dataset_name == 'flickr30k':
        required_paths = [
            data_root / 'flickr30k' / 'flickr30k_images',
            data_root / 'flickr30k' / 'flickr30k_annotations' / 'train.json',
        ]
    elif dataset_name == 'coco':
        required_paths = [
            data_root / 'coco' / 'train2017',
            data_root / 'coco' / 'val2017',
            data_root / 'coco' / 'annotations' / 'captions_train2017.json',
        ]
    else:
        return False
    
    return all(p.exists() for p in required_paths)

