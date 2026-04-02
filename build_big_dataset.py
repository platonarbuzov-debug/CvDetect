import os
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from imagededup.methods import PHash


# =========================
# CONFIG
# =========================

BASE_DIR = Path("dataset-vehicles")
RAW_DIR = BASE_DIR / "raw_downloads"
FINAL_DIR = BASE_DIR / "images" / "train"

# Сколько качать на каждый запрос из каждого источника
PER_QUERY_GOOGLE = 250
PER_QUERY_BING = 250

# Потоки: можно увеличить, если интернет и машина тянут
CRAWLER_KWARGS = dict(
    feeder_threads=2,
    parser_threads=4,
    downloader_threads=8,
)

# Минимальный размер картинки, чтобы отсечь совсем мусор
MIN_WIDTH = 320
MIN_HEIGHT = 240

# Порог дедупликации: меньше -> агрессивнее
PHASH_MAX_DISTANCE = 8


# =========================
# CLASS QUERIES
# =========================

CLASS_QUERIES: Dict[str, List[str]] = {
    "Car": [
        "car street photo",
        "sedan road photo",
        "car traffic real photo",
        "vehicle road real world"
    ],
    "Motorcycle": [
        "motorcycle street photo",
        "motorbike road photo",
        "motorcycle real world",
        "bike rider street photo"
    ],
    "Truck": [
        "cargo truck road photo",
        "lorry highway photo",
        "delivery truck real photo",
        "freight truck street"
    ],
    "Bus": [
        "city bus street photo",
        "public bus road photo",
        "coach bus real photo",
        "bus traffic photo"
    ],
    "Bicycle": [
        "bicycle road photo",
        "bike street photo",
        "cyclist road real photo",
        "bicycle urban photo"
    ],
    "Airplane": [
        "military airplane photo",
        "fighter jet photo",
        "aircraft runway photo",
        "military aircraft real photo"
    ],
    "Helicopter": [
        "military helicopter photo",
        "helicopter landing photo",
        "combat helicopter real photo",
        "helicopter airborne photo"
    ],
    "Person": [
        "person walking street photo",
        "human full body outdoor photo",
        "person standing real photo",
        "people outdoor real world"
    ],
    "Bunker": [
        "military bunker photo",
        "fortification bunker real photo",
        "defensive position bunker photo",
        "concrete bunker military photo"
    ],
    "Tank": [
        "military tank photo",
        "battle tank real photo",
        "tank field photo",
        "armored tank real world"
    ],
    "IFV": [
        "infantry fighting vehicle photo",
        "ifv military vehicle photo",
        "armored fighting vehicle real photo",
        "bmp military vehicle photo"
    ],
    "MLRS": [
        "multiple launch rocket system photo",
        "mlrs military vehicle photo",
        "rocket artillery launcher photo",
        "grad smerch himars launcher photo"
    ],
    "Cannon": [
        "artillery cannon photo",
        "howitzer gun photo",
        "field artillery real photo",
        "towed artillery cannon photo"
    ],
}


# =========================
# HELPERS
# =========================

def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_QUERIES:
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)
        (FINAL_DIR / cls).mkdir(parents=True, exist_ok=True)


def crawl_google(class_name: str, query: str, out_dir: Path) -> None:
    crawler = GoogleImageCrawler(
        storage={"root_dir": str(out_dir)},
        **CRAWLER_KWARGS,
    )
    filters = dict(size="large", type="photo")
    crawler.crawl(
        keyword=query,
        filters=filters,
        max_num=PER_QUERY_GOOGLE,
        overwrite=False,
    )


def crawl_bing(class_name: str, query: str, out_dir: Path) -> None:
    crawler = BingImageCrawler(
        storage={"root_dir": str(out_dir)},
        **CRAWLER_KWARGS,
    )
    filters = dict(size="large", type="photo", layout="wide")
    crawler.crawl(
        keyword=query,
        filters=filters,
        max_num=PER_QUERY_BING,
        overwrite=False,
    )


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            width, height = img.size
        return width >= MIN_WIDTH and height >= MIN_HEIGHT
    except Exception:
        return False


def remove_invalid_images(folder: Path) -> int:
    removed = 0
    for path in folder.rglob("*"):
        if path.is_file():
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
                continue

            if not is_valid_image(path):
                try:
                    path.unlink()
                    removed += 1
                except Exception:
                    pass
    return removed


def flatten_class_folder(raw_class_dir: Path, final_class_dir: Path) -> int:
    """
    Собирает все картинки класса из вложенных папок в одну итоговую папку.
    """
    moved = 0
    idx = 0
    for path in raw_class_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue

        ext = ".jpg" if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} else path.suffix.lower()
        target = final_class_dir / f"{raw_class_dir.name}_{idx:06d}{ext}"
        while target.exists():
            idx += 1
            target = final_class_dir / f"{raw_class_dir.name}_{idx:06d}{ext}"

        try:
            shutil.copy2(path, target)
            moved += 1
            idx += 1
        except Exception:
            pass
    return moved


def deduplicate_folder(folder: Path, max_distance_threshold: int = PHASH_MAX_DISTANCE) -> int:
    """
    Удаляет визуальные дубликаты в рамках одной папки класса.
    Оставляет первый файл, найденные дубли удаляет.
    """
    phasher = PHash()
    encodings = phasher.encode_images(image_dir=str(folder))
    duplicates = phasher.find_duplicates(
        encoding_map=encodings,
        max_distance_threshold=max_distance_threshold,
    )

    to_remove = set()
    for original, dup_list in duplicates.items():
        for dup_name in dup_list:
            if dup_name != original:
                to_remove.add(dup_name)

    removed = 0
    for name in to_remove:
        path = folder / name
        if path.exists():
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def count_images(folder: Path) -> int:
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )


# =========================
# MAIN
# =========================

def main() -> None:
    ensure_dirs()

    print("\n=== STEP 1: DOWNLOAD ===")
    for class_name, queries in CLASS_QUERIES.items():
        class_raw_dir = RAW_DIR / class_name
        class_raw_dir.mkdir(parents=True, exist_ok=True)

        for i, query in enumerate(queries, start=1):
            google_dir = class_raw_dir / f"google_q{i}"
            bing_dir = class_raw_dir / f"bing_q{i}"
            google_dir.mkdir(parents=True, exist_ok=True)
            bing_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{class_name}] Google: {query}")
            try:
                crawl_google(class_name, query, google_dir)
            except Exception as e:
                print(f"  Google error: {e}")

            print(f"[{class_name}] Bing:   {query}")
            try:
                crawl_bing(class_name, query, bing_dir)
            except Exception as e:
                print(f"  Bing error: {e}")

    print("\n=== STEP 2: REMOVE INVALID FILES ===")
    total_removed_invalid = 0
    for class_name in CLASS_QUERIES:
        removed = remove_invalid_images(RAW_DIR / class_name)
        total_removed_invalid += removed
        print(f"[{class_name}] removed invalid/non-image files: {removed}")

    print("\n=== STEP 3: FLATTEN INTO FINAL CLASS FOLDERS ===")
    for class_name in CLASS_QUERIES:
        moved = flatten_class_folder(RAW_DIR / class_name, FINAL_DIR / class_name)
        print(f"[{class_name}] collected into final folder: {moved}")

    print("\n=== STEP 4: DEDUPLICATE PER CLASS ===")
    for class_name in CLASS_QUERIES:
        class_final_dir = FINAL_DIR / class_name
        before = count_images(class_final_dir)
        removed = deduplicate_folder(class_final_dir)
        after = count_images(class_final_dir)
        print(f"[{class_name}] before={before}, removed_duplicates={removed}, after={after}")

    print("\n=== DONE ===")
    print(f"Raw downloads: {RAW_DIR}")
    print(f"Final dataset: {FINAL_DIR}")
    print("\nCounts per class:")
    for class_name in CLASS_QUERIES:
        n = count_images(FINAL_DIR / class_name)
        print(f"  {class_name}: {n}")


if __name__ == "__main__":
    main()