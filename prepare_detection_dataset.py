import random
import shutil
from pathlib import Path

random.seed(42)

# УКАЖИ ПРАВИЛЬНЫЙ ИСТОЧНИК:
# если у тебя классы лежат в dataset-vehicles/images/train/<Class>, оставь так
SRC_DIR = Path("dataset-vehicles/images/train")

# если на самом деле классы лежат в raw_downloads, замени на:
# SRC_DIR = Path("dataset-vehicles/raw_downloads")

DST_DIR = Path("dataset-vehicles/detect")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_images():
    files = []
    for class_dir in SRC_DIR.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                files.append((class_dir.name, img_path))
    return files


def split_dataset(files):
    random.shuffle(files)
    n = len(files)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def copy_files(split_map):
    counters = {"train": 0, "val": 0, "test": 0}

    for split, items in split_map.items():
        for class_name, src_path in items:
            idx = counters[split]
            new_name = f"{class_name}_{idx:06d}{src_path.suffix.lower()}"
            dst_path = DST_DIR / "images" / split / new_name
            shutil.copy2(src_path, dst_path)
            counters[split] += 1


def main():
    ensure_dirs()
    files = collect_images()
    print(f"Всего изображений найдено: {len(files)}")

    split_map = split_dataset(files)
    for split, items in split_map.items():
        print(f"{split}: {len(items)}")

    copy_files(split_map)
    print("Готово. Detection-датасет собран в dataset-vehicles/detect")


if __name__ == "__main__":
    main()