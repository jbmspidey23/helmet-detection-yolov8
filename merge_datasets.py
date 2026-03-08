"""
Merge all helmet detection datasets into a single unified dataset.

Sources:
  1. Kaggle BikesHelmets (764 images) - already in PyCharm project
     Classes: 0=helmet, 1=no-helmet (already correct)

  2. Roboflow Hard Hat Workers (~5,188 images) - in HelmetDetection project
     Classes: 0=head, 1=helmet, 2=person
     Remap: 1->0 (helmet), 0->1 (head->no-helmet), 2->DROP (person)

Output:
  - Unified dataset in datasets/helmet_merged/ with train/valid/test splits
  - Classes: 0=helmet, 1=no-helmet
  - Split: 70% train, 20% valid, 10% test
"""

import os
import shutil
import random
from pathlib import Path


# === CONFIGURATION ===
PYCHARM_PROJECT = r"C:\Users\jebin\PycharmProjects\PythonProject1"
HELMET_DETECTION = r"C:\Users\jebin\.gemini\antigravity\scratch\HelmetDetection"

# Source datasets
KAGGLE_IMAGES = os.path.join(PYCHARM_PROJECT, "datasets", "helmet_data", "images")
KAGGLE_LABELS = os.path.join(PYCHARM_PROJECT, "datasets", "helmet_data", "labels")

ROBOFLOW_TRAIN_IMAGES = os.path.join(HELMET_DETECTION, "datasets", "train", "images")
ROBOFLOW_TRAIN_LABELS = os.path.join(HELMET_DETECTION, "datasets", "train", "labels")
ROBOFLOW_TEST_IMAGES = os.path.join(HELMET_DETECTION, "datasets", "test", "images")
ROBOFLOW_TEST_LABELS = os.path.join(HELMET_DETECTION, "datasets", "test", "labels")

# Output
OUTPUT_BASE = os.path.join(PYCHARM_PROJECT, "datasets", "helmet_merged")

# Split ratios
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42


def remap_roboflow_label(label_path, output_path):
    """
    Remap Roboflow label classes:
      0 (head)    -> 1 (no-helmet)
      1 (helmet)  -> 0 (helmet)
      2 (person)  -> DROP
    Returns True if at least one valid annotation remains.
    """
    remapped_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            if cls == 1:       # helmet -> 0
                parts[0] = "0"
                remapped_lines.append(" ".join(parts))
            elif cls == 0:     # head -> 1 (no-helmet)
                parts[0] = "1"
                remapped_lines.append(" ".join(parts))
            # cls == 2 (person) -> skip

    if remapped_lines:
        with open(output_path, "w") as f:
            f.write("\n".join(remapped_lines) + "\n")
        return True
    return False


def copy_kaggle_label(label_path, output_path):
    """Copy Kaggle labels as-is (classes already correct: 0=helmet, 1=no-helmet)."""
    shutil.copy2(label_path, output_path)
    return True


def collect_dataset():
    """Collect all image-label pairs from all sources."""
    pairs = []  # list of (image_path, label_path, source)

    # 1. Kaggle dataset
    if os.path.exists(KAGGLE_IMAGES):
        for img_file in os.listdir(KAGGLE_IMAGES):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(KAGGLE_IMAGES, img_file)
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(KAGGLE_LABELS, lbl_file)
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    pairs.append((img_path, lbl_path, "kaggle"))

    # 2. Roboflow train
    if os.path.exists(ROBOFLOW_TRAIN_IMAGES):
        for img_file in os.listdir(ROBOFLOW_TRAIN_IMAGES):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(ROBOFLOW_TRAIN_IMAGES, img_file)
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(ROBOFLOW_TRAIN_LABELS, lbl_file)
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    pairs.append((img_path, lbl_path, "roboflow"))

    # 3. Roboflow test
    if os.path.exists(ROBOFLOW_TEST_IMAGES):
        for img_file in os.listdir(ROBOFLOW_TEST_IMAGES):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(ROBOFLOW_TEST_IMAGES, img_file)
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                lbl_path = os.path.join(ROBOFLOW_TEST_LABELS, lbl_file)
                if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
                    pairs.append((img_path, lbl_path, "roboflow"))

    return pairs


def main():
    print("=" * 60)
    print("  Helmet Detection Dataset Merger")
    print("=" * 60)

    # Collect all pairs
    pairs = collect_dataset()
    print(f"\nTotal image-label pairs found: {len(pairs)}")

    kaggle_count = sum(1 for _, _, s in pairs if s == "kaggle")
    roboflow_count = sum(1 for _, _, s in pairs if s == "roboflow")
    print(f"  Kaggle:   {kaggle_count}")
    print(f"  Roboflow: {roboflow_count}")

    # Shuffle
    random.seed(SEED)
    random.shuffle(pairs)

    # Split
    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    splits = {
        "train": pairs[:train_end],
        "valid": pairs[train_end:valid_end],
        "test": pairs[valid_end:],
    }

    # Create output directories
    for split_name in splits:
        os.makedirs(os.path.join(OUTPUT_BASE, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, split_name, "labels"), exist_ok=True)

    # Process each split
    stats = {"helmet": 0, "no_helmet": 0, "dropped_person": 0}

    for split_name, split_pairs in splits.items():
        processed = 0
        skipped = 0

        for img_path, lbl_path, source in split_pairs:
            img_filename = os.path.basename(img_path)
            lbl_filename = os.path.splitext(img_filename)[0] + ".txt"

            dst_img = os.path.join(OUTPUT_BASE, split_name, "images", img_filename)
            dst_lbl = os.path.join(OUTPUT_BASE, split_name, "labels", lbl_filename)

            if source == "kaggle":
                shutil.copy2(img_path, dst_img)
                copy_kaggle_label(lbl_path, dst_lbl)
                processed += 1
            elif source == "roboflow":
                success = remap_roboflow_label(lbl_path, dst_lbl)
                if success:
                    shutil.copy2(img_path, dst_img)
                    processed += 1
                else:
                    skipped += 1  # All annotations were "person" -> nothing left

        print(f"\n{split_name}: {processed} images copied, {skipped} skipped (no valid labels)")

    # Count class distribution in the merged dataset
    print("\n" + "=" * 60)
    print("  Class Distribution in Merged Dataset")
    print("=" * 60)

    for split_name in ["train", "valid", "test"]:
        labels_dir = os.path.join(OUTPUT_BASE, split_name, "labels")
        helmet_count = 0
        no_helmet_count = 0
        for lbl_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, lbl_file), "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls = int(line.split()[0])
                    if cls == 0:
                        helmet_count += 1
                    elif cls == 1:
                        no_helmet_count += 1
        print(f"  {split_name}: {helmet_count} helmet, {no_helmet_count} no-helmet")

    # Create data.yaml for the merged dataset
    data_yaml = os.path.join(PYCHARM_PROJECT, "data_merged.yaml")
    with open(data_yaml, "w") as f:
        f.write(f"train: datasets/helmet_merged/train/images\n")
        f.write(f"val: datasets/helmet_merged/valid/images\n")
        f.write(f"test: datasets/helmet_merged/test/images\n")
        f.write(f"\n")
        f.write(f"nc: 2\n")
        f.write(f"\n")
        f.write(f"names:\n")
        f.write(f"  0: helmet\n")
        f.write(f"  1: no-helmet\n")

    print(f"\ndata_merged.yaml created at: {data_yaml}")
    print("\nDone! Run training with:")
    print('  python train.py')


if __name__ == "__main__":
    main()
