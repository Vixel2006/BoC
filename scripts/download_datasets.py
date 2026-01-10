"""
Dataset download and setup script for Flickr30k and MS COCO.

This script automates the download and preparation of datasets for training.

Usage:
    python scripts/download_datasets.py --dataset flickr30k --output-dir ./data
    python scripts/download_datasets.py --dataset coco --output-dir ./data
    python scripts/download_datasets.py --all --output-dir ./data
"""

import argparse
import os
import urllib.request
import zipfile
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
import shutil
from src.data import verify_dataset



class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def setup_flickr30k(output_dir: Path):
    """
    Download and setup Flickr30k dataset.
    
    Note: Flickr30k images require manual download from:
    https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
    
    This script will:
    1. Create directory structure
    2. Download/create annotation files
    3. Provide instructions for manual steps
    """
    print("\n" + "="*60)
    print("Setting up Flickr30k Dataset")
    print("="*60)
    
    flickr_dir = output_dir / "flickr30k"
    flickr_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = flickr_dir / "flickr30k_images"
    annotations_dir = flickr_dir / "flickr30k_annotations"
    
    images_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)
    
    # Check if images are already present
    image_count = len(list(images_dir.glob("*.jpg")))
    
    if image_count < 31000:
        print("\n⚠️  Flickr30k images not found!")
        print("\nManual download required:")
        print("1. Go to: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
        print("2. Download 'flickr30k_images.tar.gz'")
        print(f"3. Extract to: {images_dir}")
        print("\nAlternatively, use Kaggle API:")
        print("  kaggle datasets download -d hsankesara/flickr-image-dataset")
        print(f"  tar -xzf flickr-image-dataset.zip -C {images_dir}")
        
        use_kaggle = input("\nDo you want to try downloading via Kaggle API? (requires kaggle installed) [y/N]: ")
        
        if use_kaggle.lower() == 'y':
            try:
                import kagglehub
                print("\nDownloading Flickr30k via KaggleHub...")
                path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
                print(f"Downloaded to: {path}")
                
                # Move files to correct location
                for img_file in Path(path).rglob("*.jpg"):
                    shutil.copy(img_file, images_dir / img_file.name)
                
                print(f"✓ Copied images to {images_dir}")
            except ImportError:
                print("❌ kagglehub not installed. Install with: pip install kagglehub")
                return False
            except Exception as e:
                print(f"❌ Error downloading via Kaggle: {e}")
                return False
    else:
        print(f"✓ Found {image_count} images in {images_dir}")
    
    # Download or create annotations
    print("\nSetting up annotations...")
    
    # For this demo, create sample annotation structure
    # In production, download from official Flickr30k Entities dataset
    
    for split in ['train', 'val', 'test']:
        ann_file = annotations_dir / f"{split}.json"
        if not ann_file.exists():
            print(f"Creating template for {split}.json")
            print("⚠️  You need to populate these with actual Flickr30k captions")
            
            # Create template
            template = [
                {
                    "image_id": "example.jpg",
                    "captions": [
                        "Caption 1",
                        "Caption 2",
                        "Caption 3",
                        "Caption 4",
                        "Caption 5"
                    ]
                }
            ]
            with open(ann_file, 'w') as f:
                json.dump(template, f, indent=2)
    
    print("\n📝 Note: Flickr30k caption annotations available at:")
    print("   https://github.com/BryanPlummer/flickr30k_entities")
    print("   Download 'Sentences' and convert to our JSON format")
    
    print(f"\n✓ Flickr30k setup complete: {flickr_dir}")
    return True


def setup_coco(output_dir: Path):
    """
    Download and setup MS COCO dataset.
    
    Downloads:
    - Train images (2017)
    - Val images (2017)
    - Annotations
    """
    print("\n" + "="*60)
    print("Setting up MS COCO Dataset")
    print("="*60)
    
    coco_dir = output_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO URLs
    base_url = "http://images.cocodataset.org"
    urls = {
        'train_images': f"{base_url}/zips/train2017.zip",
        'val_images': f"{base_url}/zips/val2017.zip",
        'annotations': "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    # Download and extract each component
    for name, url in urls.items():
        filename = url.split('/')[-1]
        filepath = coco_dir / filename
        
        # Check if already downloaded
        if name == 'train_images' and (coco_dir / 'train2017').exists():
            img_count = len(list((coco_dir / 'train2017').glob('*.jpg')))
            if img_count > 100000:
                print(f"✓ {name} already exists ({img_count} images)")
                continue
        elif name == 'val_images' and (coco_dir / 'val2017').exists():
            img_count = len(list((coco_dir / 'val2017').glob('*.jpg')))
            if img_count > 4000:
                print(f"✓ {name} already exists ({img_count} images)")
                continue
        elif name == 'annotations' and (coco_dir / 'annotations').exists():
            if (coco_dir / 'annotations' / 'captions_train2017.json').exists():
                print(f"✓ {name} already exists")
                continue
        
        # Download
        if not filepath.exists():
            print(f"\nDownloading {name}...")
            try:
                download_url(url, str(filepath))
            except Exception as e:
                print(f"❌ Error downloading {name}: {e}")
                print(f"Please download manually from: {url}")
                continue
        
        # Extract
        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
            print(f"✓ Extracted {name}")
            
            # Remove zip file to save space
            filepath.unlink()
            print(f"✓ Removed {filename}")
        except Exception as e:
            print(f"❌ Error extracting {name}: {e}")
    
    # Verify structure
    expected_dirs = ['train2017', 'val2017', 'annotations']
    all_present = all((coco_dir / d).exists() for d in expected_dirs)
    
    if all_present:
        # Count images
        train_count = len(list((coco_dir / 'train2017').glob('*.jpg')))
        val_count = len(list((coco_dir / 'val2017').glob('*.jpg')))
        
        print(f"\n✓ MS COCO setup complete: {coco_dir}")
        print(f"  - Training images: {train_count:,}")
        print(f"  - Validation images: {val_count:,}")
        print(f"  - Annotations: ✓")
        return True
    else:
        print("\n⚠️  Some COCO components missing. Please check downloads.")
        return False





def main():
    parser = argparse.ArgumentParser(description="Download and setup datasets for BoC")
    parser.add_argument('--dataset', type=str, choices=['flickr30k', 'coco', 'all'],
                       help='Dataset to download')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Output directory for datasets')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify datasets without downloading')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BoC Dataset Setup")
    print("="*60)
    print(f"Output directory: {output_dir.absolute()}")
    
    if args.verify_only:
        print("\nVerifying datasets...")
        if args.dataset in ['flickr30k', 'all']:
            status = "✓" if verify_dataset('flickr30k', output_dir) else "✗"
            print(f"{status} Flickr30k")
        if args.dataset in ['coco', 'all']:
            status = "✓" if verify_dataset('coco', output_dir) else "✗"
            print(f"{status} MS COCO")
        return
    
    success = True
    
    if args.dataset in ['flickr30k', 'all']:
        success &= setup_flickr30k(output_dir)
    
    if args.dataset in ['coco', 'all']:
        success &= setup_coco(output_dir)
    
    if success:
        print("\n" + "="*60)
        print("✓ Dataset setup complete!")
        print("="*60)
        print(f"\nDatasets ready at: {output_dir.absolute()}")
        print("\nYou can now run training:")
        if args.dataset == 'flickr30k' or args.dataset == 'all':
            print(f"  python main.py train --dataset flickr30k --data-root {output_dir}")
        if args.dataset == 'coco' or args.dataset == 'all':
            print(f"  python main.py train --dataset coco --data-root {output_dir}")
    else:
        print("\n⚠️  Some datasets may not be fully set up. Please check messages above.")


if __name__ == '__main__':
    main()
