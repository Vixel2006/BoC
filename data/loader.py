import os
import torch
from torchvision.datasets import Flickr30k
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Callable

def flickr30k_loader(
    root: str = "/flickr30k",
    split: str = "train", # Added split argument
    transforms: Callable = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
):
    ann_file = os.path.join(root, "annotations", f"captions_{split}.txt") # Dynamically construct ann_file
    dataset = Flickr30k(root=root, ann_file=ann_file, transform=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader

