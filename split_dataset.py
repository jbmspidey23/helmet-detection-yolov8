"""
Split the helmet dataset into train/valid/test sets.
Copies images and labels from the flat folders into the YOLO directory structure.
Split: 70% train, 20% valid, 10% test
"""

import os
import shutil
import random

# Paths
BASE = os.path.join("datasets", "helmet_data")
IMAGES_DIR = os.path.join(BASE, "images")
LABELS_DIR = os.path.join(BASE, "labels")

SPLITS = {
    "train": 0.7,
    "valid": 0.2,
    "test": 0.1,
}

def main():
    # Get all image files
    all_images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Total images found: {len(all_images)}")

    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_images)

    # Calculate split sizes
    total = len(all_images)
    train_end = int(total * SPLITS["train"])
    valid_end = train_end + int(total * SPLITS["valid"])

    splits = {
        "train": all_images[:train_end],
        "valid": all_images[train_end:valid_end],
        "test": all_images[valid_end:],
    }

    for split_name, files in splits.items():
        img_dst = os.path.join(BASE, split_name, "images")
        lbl_dst = os.path.join(BASE, split_name, "labels")
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        copied = 0
        skipped = 0
        for img_file in files:
            # Copy image
            src_img = os.path.join(IMAGES_DIR, img_file)
            shutil.copy2(src_img, os.path.join(img_dst, img_file))

            # Copy matching label
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_lbl = os.path.join(LABELS_DIR, label_file)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dst, label_file))
                copied += 1
            else:
                skipped += 1

        print(f"{split_name}: {len(files)} images, {copied} labels copied, {skipped} labels missing")

    print("\nDone! Dataset is ready for training.")


if __name__ == "__main__":
    main()
