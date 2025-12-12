import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Define paths based on loader.py's defaults
FLICKR_ROOT = "/flickr30k"
IMAGE_DIR = os.path.join(FLICKR_ROOT, "flickr30k-images")
ANNOTATION_DIR = os.path.join(FLICKR_ROOT, "annotations")

SPLITS = ["train", "test"] # Also "validation" is available, but we'll stick to train/test for now

def check_data_exists():
    """Checks if the dataset (images and annotation files) already exists."""
    if not os.path.exists(IMAGE_DIR) or not os.listdir(IMAGE_DIR):
        return False

    for split in SPLITS:
        ann_file_path = os.path.join(ANNOTATION_DIR, f"captions_{{split}}.txt")
        if not os.path.exists(ann_file_path):
            return False
    return True

def download_and_prepare_flickr30k():
    if check_data_exists():
        print(f"Flickr30k dataset already exists at {FLICKR_ROOT}. Skipping download and preparation.")
        return

    # Create directories if they don't exist
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    print("Downloading and preparing nlphuji/flickr30k dataset...")

    for split in SPLITS:
        print(f"Processing {split} split...")
        dataset = load_dataset("nlphuji/flickr30k", split=split)
        ann_file_path = os.path.join(ANNOTATION_DIR, f"captions_{{split}}.txt")

        with open(ann_file_path, "w", encoding="utf-8") as f_ann:
            for item in tqdm(dataset, desc=f"Saving {split} images and annotations"):
                image = item["image"]
                image_id = item["img_id"]
                captions = item["sentences"]["raw"]

                # Save image
                image_filename = f"{image_id}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                
                # Only save image if it doesn't exist to avoid overwriting
                if not os.path.exists(image_path):
                    image.save(image_path)

                # Write captions to annotation file
                for caption in captions:
                    f_ann.write(f"{image_filename}\t{caption}\n")

        print(f"{split} split prepared. Annotations saved to {ann_file_path}")

    print(f"Flickr30k dataset prepared at {FLICKR_ROOT}")
    print(f"Images saved to {IMAGE_DIR}")
    print(f"Annotations saved to {ANNOTATION_DIR}")

if __name__ == "__main__":
    download_and_prepare_flickr30k()
