from pathlib import Path

DATASET_DIR = Path("dataset-vehicles/detect")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

def has_image(image_dir: Path, stem: str) -> bool:
    return any((image_dir / f"{stem}{ext}").exists() for ext in IMAGE_EXTS)

def clean_split(split: str):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    removed = 0

    for label_path in label_dir.glob("*.txt"):
        if not has_image(image_dir, label_path.stem):
            print(f"remove orphan label: {label_path}")
            label_path.unlink()
            removed += 1

    print(f"{split}: removed {removed}")

for split in ["train", "val", "test"]:
    clean_split(split)