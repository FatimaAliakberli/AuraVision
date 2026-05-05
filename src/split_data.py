"""
split_data.py
=============
Splits the downloaded crowd_dataset/ into train/ and test/ subsets
using an 80/20 ratio, preserving the class subfolder structure so
PyTorch's ImageFolder can load it directly.

Run after data_extraction.py.
"""

import os
import shutil
import random

# ── Configuration ──────────────────────────────────────────────────────────────
SOURCE_DIR   = "extracted_data"    # output of data_extraction.py
OUTPUT_DIR   = "dataset"   # destination: split_dataset/train/ & /test/
SPLIT_RATIO  = 0.8               # 80% train, 20% test
RANDOM_SEED  = 42
# ──────────────────────────────────────────────────────────────────────────────


def split_dataset(source_dir: str, output_dir: str, split_ratio: float = 0.8) -> None:
    """
    Copies images from source_dir/<Class>/ into:
        output_dir/train/<Class>/
        output_dir/test/<Class>/
    using a reproducible random shuffle.
    """
    random.seed(RANDOM_SEED)

    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
    ]

    # Create destination folders up front
    for subset in ("train", "test"):
        for cls in classes:
            os.makedirs(os.path.join(output_dir, subset, cls), exist_ok=True)

    total_train = total_test = 0

    for class_name in sorted(classes):
        class_path = os.path.join(source_dir, class_name)
        images = sorted([
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        random.shuffle(images)

        split_point  = int(len(images) * split_ratio)
        train_images = images[:split_point]
        test_images  = images[split_point:]

        for fname in train_images:
            shutil.copy(
                os.path.join(class_path, fname),
                os.path.join(output_dir, "train", class_name, fname),
            )
        for fname in test_images:
            shutil.copy(
                os.path.join(class_path, fname),
                os.path.join(output_dir, "test", class_name, fname),
            )

        print(f"  {class_name:<12}  train: {len(train_images):>4}  |  test: {len(test_images):>4}")
        total_train += len(train_images)
        total_test  += len(test_images)

    print(f"\n  ── Totals ──  train: {total_train}  |  test: {total_test}")
    print(f"\n✅ Split complete → {output_dir}/")


if __name__ == "__main__":
    print(f"Splitting '{SOURCE_DIR}' into '{OUTPUT_DIR}' ({int(SPLIT_RATIO*100)}/{int((1-SPLIT_RATIO)*100)}) ...\n")
    split_dataset(SOURCE_DIR, OUTPUT_DIR, SPLIT_RATIO)
