import json
import shutil
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from imagededup.methods import PHash

# =========================
# CONFIG
# =========================
BASE_DIR = Path("dataset-vehicles")
RUN_ID = time.strftime("%Y%m%d_%H%M%S")

RAW_DIR = BASE_DIR / "raw_downloads" / f"topup_{RUN_ID}"
FINAL_DIR = BASE_DIR / "images" / "train"
REPORTS_DIR = BASE_DIR / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Цель: добрать примерно 2-3k изображений суммарно
TARGET_EXTRA_PER_CLASS: Dict[str, int] = {
    "Car": 250,
    "Motorcycle": 180,
    "Truck": 220,
    "Bus": 180,
    "Bicycle": 180,
    "Airplane": 180,
    "Helicopter": 160,
    "Person": 320,
    "Bunker": 140,
    "Tank": 240,
    "IFV": 200,
    "MLRS": 140,
    "Cannon": 170,
}

PER_QUERY_GOOGLE = 120
PER_QUERY_BING = 120

CRAWLER_KWARGS = dict(
    feeder_threads=2,
    parser_threads=4,
    downloader_threads=8,
)

MIN_WIDTH = 320
MIN_HEIGHT = 240
PHASH_MAX_DISTANCE = 8

CLASS_QUERIES: Dict[str, List[str]] = {
    "Car": [
        "car street photo",
        "sedan road photo",
        "hatchback city street photo",
        "car traffic real photo",
        "parked car street real photo",
        "vehicle road real world",
    ],
    "Motorcycle": [
        "motorcycle street photo",
        "motorbike road photo",
        "motorcycle real world",
        "bike rider street photo",
        "urban motorcycle photo",
        "parked motorcycle street",
    ],
    "Truck": [
        "cargo truck road photo",
        "lorry highway photo",
        "delivery truck real photo",
        "freight truck street",
        "box truck city photo",
        "semi truck road real photo",
    ],
    "Bus": [
        "city bus street photo",
        "public bus road photo",
        "coach bus real photo",
        "bus traffic photo",
        "urban bus stop photo",
        "passenger bus road photo",
    ],
    "Bicycle": [
        "bicycle road photo",
        "bike street photo",
        "cyclist road real photo",
        "bicycle urban photo",
        "parked bicycle street",
        "city bicycle real world",
    ],
    "Airplane": [
        "military airplane photo",
        "fighter jet photo",
        "aircraft runway photo",
        "military aircraft real photo",
        "warplane landing photo",
        "combat aircraft airfield photo",
    ],
    "Helicopter": [
        "military helicopter photo",
        "helicopter landing photo",
        "combat helicopter real photo",
        "helicopter airborne photo",
        "military helicopter runway photo",
        "helicopter tarmac photo",
    ],
    "Person": [
        "person walking street photo",
        "human full body outdoor photo",
        "person standing real photo",
        "people outdoor real world",
        "pedestrian street full body photo",
        "single person outdoor photo",
    ],
    "Bunker": [
        "military bunker photo",
        "fortification bunker real photo",
        "defensive position bunker photo",
        "concrete bunker military photo",
        "pillbox bunker exterior photo",
        "field bunker military position",
    ],
    "Tank": [
        "military tank photo",
        "battle tank real photo",
        "tank field photo",
        "armored tank real world",
        "tank driving military photo",
        "main battle tank real photo",
    ],
    "IFV": [
        "infantry fighting vehicle photo",
        "ifv military vehicle photo",
        "armored fighting vehicle real photo",
        "bmp military vehicle photo",
        "tracked ifv real photo",
        "combat vehicle ifv photo",
    ],
    "MLRS": [
        "multiple launch rocket system photo",
        "mlrs military vehicle photo",
        "rocket artillery launcher photo",
        "grad smerch himars launcher photo",
        "rocket launcher truck military photo",
        "mlrs field photo",
    ],
    "Cannon": [
        "artillery cannon photo",
        "howitzer gun photo",
        "field artillery real photo",
        "towed artillery cannon photo",
        "artillery piece outdoor photo",
        "howitzer military position photo",
    ],
}


# =========================
# HELPERS
# =========================
def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for cls in CLASS_QUERIES:
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)
        (FINAL_DIR / cls).mkdir(parents=True, exist_ok=True)


def count_images(folder: Path) -> int:
    return sum(
        1
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
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
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
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


def next_index_for_class(final_class_dir: Path, class_name: str) -> int:
    prefix = f"{class_name.lower()}_"
    max_idx = -1
    for p in final_class_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem.lower()
        if stem.startswith(prefix):
            tail = stem[len(prefix):]
            if tail.isdigit():
                max_idx = max(max_idx, int(tail))
    return max_idx + 1


def copy_images_from_query_dir(query_dir: Path, final_class_dir: Path, class_name: str) -> int:
    copied = 0
    idx = next_index_for_class(final_class_dir, class_name)

    for path in sorted(query_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        ext = path.suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"

        target = final_class_dir / f"{class_name.lower()}_{idx:06d}{ext}"
        while target.exists():
            idx += 1
            target = final_class_dir / f"{class_name.lower()}_{idx:06d}{ext}"

        try:
            shutil.copy2(path, target)
            copied += 1
            idx += 1
        except Exception:
            pass

    return copied


def deduplicate_folder(folder: Path, max_distance_threshold: int = PHASH_MAX_DISTANCE) -> int:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(images) < 2:
        return 0

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


def crawl_google(query: str, out_dir: Path) -> None:
    crawler = GoogleImageCrawler(
        storage={"root_dir": str(out_dir)},
        **CRAWLER_KWARGS,
    )
    filters = dict(size="large", type="photo")
    crawler.crawl(keyword=query, filters=filters, max_num=PER_QUERY_GOOGLE, overwrite=False)


def crawl_bing(query: str, out_dir: Path) -> None:
    crawler = BingImageCrawler(
        storage={"root_dir": str(out_dir)},
        **CRAWLER_KWARGS,
    )
    filters = dict(size="large", type="photo", layout="wide")
    crawler.crawl(keyword=query, filters=filters, max_num=PER_QUERY_BING, overwrite=False)


# =========================
# MAIN
# =========================
def main() -> None:
    ensure_dirs()

    summary = {
        "run_id": RUN_ID,
        "raw_dir": str(RAW_DIR),
        "final_dir": str(FINAL_DIR),
        "classes": {},
    }

    print(f"\n=== TOP-UP RUN {RUN_ID} ===")
    print(f"Raw dir:   {RAW_DIR}")
    print(f"Final dir: {FINAL_DIR}")

    for class_name, queries in CLASS_QUERIES.items():
        final_class_dir = FINAL_DIR / class_name
        raw_class_dir = RAW_DIR / class_name

        before_count = count_images(final_class_dir)
        target_count = before_count + TARGET_EXTRA_PER_CLASS[class_name]

        print(f"\n[{class_name}] current={before_count} | target={target_count}")

        class_info = {
            "before_count": before_count,
            "target_count": target_count,
            "invalid_removed": 0,
            "copied_before_dedup": 0,
            "duplicates_removed": 0,
            "after_count": before_count,
        }

        for i, query in enumerate(queries, start=1):
            current_count = count_images(final_class_dir)
            if current_count >= target_count:
                print(f"[{class_name}] target reached early: {current_count}/{target_count}")
                break

            google_dir = raw_class_dir / f"google_q{i}"
            bing_dir = raw_class_dir / f"bing_q{i}"
            google_dir.mkdir(parents=True, exist_ok=True)
            bing_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{class_name}] Google: {query}")
            try:
                crawl_google(query, google_dir)
            except Exception as e:
                print(f"  Google error: {e}")

            print(f"[{class_name}] Bing:   {query}")
            try:
                crawl_bing(query, bing_dir)
            except Exception as e:
                print(f"  Bing error: {e}")

            removed_google = remove_invalid_images(google_dir)
            removed_bing = remove_invalid_images(bing_dir)
            class_info["invalid_removed"] += removed_google + removed_bing

            copied_google = copy_images_from_query_dir(google_dir, final_class_dir, class_name)
            copied_bing = copy_images_from_query_dir(bing_dir, final_class_dir, class_name)
            class_info["copied_before_dedup"] += copied_google + copied_bing

            removed_dups = deduplicate_folder(final_class_dir)
            class_info["duplicates_removed"] += removed_dups

            after_query_count = count_images(final_class_dir)
            print(
                f"[{class_name}] after query {i}: count={after_query_count} "
                f"(copied={copied_google + copied_bing}, invalid_removed={removed_google + removed_bing}, "
                f"dedup_removed={removed_dups})"
            )

        final_count = count_images(final_class_dir)
        class_info["after_count"] = final_count
        class_info["added_total"] = final_count - before_count
        summary["classes"][class_name] = class_info

        print(
            f"[{class_name}] DONE | before={before_count} after={final_count} "
            f"added={class_info['added_total']}"
        )

    summary_path = REPORTS_DIR / f"topup_summary_{RUN_ID}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== FINAL COUNTS ===")
    total_before = 0
    total_after = 0
    total_added = 0
    for class_name, info in summary["classes"].items():
        total_before += info["before_count"]
        total_after += info["after_count"]
        total_added += info["added_total"]
        print(f"{class_name}: before={info['before_count']} after={info['after_count']} added={info['added_total']}")

    print(f"\nTOTAL BEFORE: {total_before}")
    print(f"TOTAL AFTER:  {total_after}")
    print(f"TOTAL ADDED:  {total_added}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
