from pathlib import Path

DATASET_DIR = Path("dataset-vehicles/detect")

def clean_split(split: str):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    removed = 0

    for label_path in label_dir.glob("*.txt"):
        if label_path.stat().st_size == 0:
            image_stem = label_path.stem

            # удалить пустой label
            label_path.unlink()

            # удалить соответствующее изображение
            found_image = False
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                img_path = image_dir / f"{image_stem}{ext}"
                if img_path.exists():
                    img_path.unlink()
                    found_image = True
                    break

            removed += 1

    print(f"{split}: удалено {removed} пустых пар image+label")

def main():
    for split in ["train", "val", "test"]:
        clean_split(split)

if __name__ == "__main__":
    main()