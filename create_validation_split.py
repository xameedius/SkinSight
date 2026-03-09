import os
import shutil
import random
from pathlib import Path

random.seed(42)

data_root = Path("/content/data")
train_dir = data_root / "train"
val_dir = data_root / "val"

val_split = 0.15  # 15%

val_dir.mkdir(exist_ok=True)

for class_dir in train_dir.iterdir():
    if not class_dir.is_dir():
        continue

    images = list(class_dir.glob("*"))
    random.shuffle(images)

    n_val = int(len(images) * val_split)

    val_class_dir = val_dir / class_dir.name
    val_class_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images[:n_val]:
        shutil.move(str(img_path), val_class_dir / img_path.name)

print("✅ Validation split created.")