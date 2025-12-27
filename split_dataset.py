import os
import shutil
import random

SOURCE_DIR = "origin_dataset"
TARGET_DIR = "dataset"

SPLIT_RATIO = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

MAX_IMAGES_PER_CLASS = 2000 

random.seed(42)

def split_dataset():
    for split in SPLIT_RATIO:
        os.makedirs(os.path.join(TARGET_DIR, split), exist_ok=True)

    for label in os.listdir(SOURCE_DIR):
        label_path = os.path.join(SOURCE_DIR, label)
        if not os.path.isdir(label_path):
            continue

        images = os.listdir(label_path)
        random.shuffle(images)

        # Limit images per class to MAX_IMAGES_PER_CLASS
        images = images[:MAX_IMAGES_PER_CLASS]

        total = len(images)
        train_end = int(total * SPLIT_RATIO["train"])
        val_end = train_end + int(total * SPLIT_RATIO["val"])

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split, files in splits.items():
            split_label_dir = os.path.join(TARGET_DIR, split, label)
            os.makedirs(split_label_dir, exist_ok=True)

            for file in files:
                src = os.path.join(label_path, file)
                dst = os.path.join(split_label_dir, file)
                shutil.copy(src, dst)

        print(f"Class {label}: {len(images)} images split")

    print("✅ Dataset split completed!")

if __name__ == "__main__":
    split_dataset()
