import json
import random
import shutil
from pathlib import Path
from typing import Dict, List

random.seed(42)

SRC_DIR = Path("dataset-vehicles/images/train")
DST_DIR = Path("dataset-vehicles/detect")

TRAIN_RATIO = 0.85
VAL_RATIO = 0.15
TEST_RATIO = 0.0

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESET_DST = True

REPORTS_DIR = Path("dataset-vehicles/reports")


def ensure_dirs():
    for split in ["train", "val", "test"]:
        (DST_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def reset_destination():
    if not RESET_DST:
        return

    for rel in [
        "images/train",
        "images/val",
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "review/train",
        "review/val",
        "review/test",
        "reports",
    ]:
        path = DST_DIR / rel
        if path.exists():
            shutil.rmtree(path)


def collect_images_by_class() -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}

    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка источника: {SRC_DIR}")

    for class_dir in sorted(SRC_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        files = [
            p for p in sorted(class_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if files:
            result[class_dir.name] = files

    return result


def split_class_files(files: List[Path]):
    files = files[:]
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


def copy_files(split_map: Dict[str, List[tuple]]):
    counters = {"train": 0, "val": 0, "test": 0}
    per_class_split = {}

    for split, items in split_map.items():
        for class_name, src_path in items:
            idx = counters[split]
            ext = src_path.suffix.lower()
            if ext == ".jpeg":
                ext = ".jpg"

            safe_class_name = class_name.replace(" ", "_")
            new_name = f"{safe_class_name}_{idx:06d}{ext}"
            dst_path = DST_DIR / "images" / split / new_name
            shutil.copy2(src_path, dst_path)
            counters[split] += 1

            per_class_split.setdefault(class_name, {"train": 0, "val": 0, "test": 0})
            per_class_split[class_name][split] += 1

    return counters, per_class_split


def main():
    reset_destination()
    ensure_dirs()

    by_class = collect_images_by_class()

    print("Найдено классов:", len(by_class))
    total = sum(len(v) for v in by_class.values())
    print("Всего изображений:", total)

    split_map = {"train": [], "val": [], "test": []}
    class_counts = {}

    for class_name, files in by_class.items():
        class_counts[class_name] = len(files)
        parts = split_class_files(files)

        split_map["train"].extend((class_name, p) for p in parts["train"])
        split_map["val"].extend((class_name, p) for p in parts["val"])
        split_map["test"].extend((class_name, p) for p in parts["test"])

        print(
            f"{class_name}: total={len(files)} "
            f"train={len(parts['train'])} "
            f"val={len(parts['val'])} "
            f"test={len(parts['test'])}"
        )

    random.shuffle(split_map["train"])
    random.shuffle(split_map["val"])
    random.shuffle(split_map["test"])

    counters, per_class_split = copy_files(split_map)

    summary = {
        "src_dir": str(SRC_DIR),
        "dst_dir": str(DST_DIR),
        "ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "totals": {
            "all": total,
            "train": counters["train"],
            "val": counters["val"],
            "test": counters["test"],
        },
        "class_counts": class_counts,
        "per_class_split": per_class_split,
    }

    summary_path = REPORTS_DIR / "prepare_detection_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nГотово.")
    print(f"train: {counters['train']}")
    print(f"val:   {counters['val']}")
    print(f"test:  {counters['test']}")
    print(f"Summary JSON: {summary_path}")
    print(f"Detection-датасет пересобран в {DST_DIR}")


if __name__ == "__main__":
    main()
