from pathlib import Path
from collections import defaultdict

from groundingdino.util.inference import load_model, load_image, predict

# ----------------------------
# CONFIG
# ----------------------------

CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

DATASET_DIR = Path("dataset-vehicles/detect")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = "cpu"

# ВАЖНО:
# GroundingDINO рекомендует разделять категории точками.
# Используем более естественные фразы для военных классов.
TEXT_PROMPT = (
    "car . motorcycle . truck . bus . bicycle . "
    "airplane . helicopter . person . bunker . tank . "
    "infantry fighting vehicle . multiple launch rocket system . artillery cannon ."
)

# YOLO class ids
CLASS_IDS = {
    "car": 0,
    "motorcycle": 1,
    "truck": 2,
    "bus": 3,
    "bicycle": 4,
    "airplane": 5,
    "helicopter": 6,
    "person": 7,
    "bunker": 8,
    "tank": 9,
    "ifv": 10,
    "mlrs": 11,
    "cannon": 12,
}

# Фразы GroundingDINO -> наши классы
ALIASES = {
    "car": "car",
    "motorcycle": "motorcycle",
    "truck": "truck",
    "bus": "bus",
    "bicycle": "bicycle",
    "airplane": "airplane",
    "plane": "airplane",
    "aircraft": "airplane",
    "helicopter": "helicopter",
    "person": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "bunker": "bunker",
    "tank": "tank",
    "infantry fighting vehicle": "ifv",
    "ifv": "ifv",
    "bmp": "ifv",
    "multiple launch rocket system": "mlrs",
    "mlrs": "mlrs",
    "rocket system": "mlrs",
    "artillery cannon": "cannon",
    "cannon": "cannon",
    "howitzer": "cannon",
    "artillery": "cannon",
}

BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.15


# ----------------------------
# HELPERS
# ----------------------------

def normalize_phrase(phrase: str) -> str | None:
    phrase = phrase.strip().lower()

    # Ищем наиболее длинные совпадения сначала
    for key in sorted(ALIASES.keys(), key=len, reverse=True):
        if key in phrase:
            return ALIASES[key]

    return None


def deduplicate_lines(lines: list[str]) -> list[str]:
    return sorted(set(lines))


def process_split(split: str, model) -> None:
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    image_files.sort()

    print(f"\nОбрабатываю {split}: {len(image_files)} изображений")

    stats = defaultdict(int)

    for idx, image_path in enumerate(image_files, start=1):
        try:
            image_source, image = load_image(str(image_path))

            # В official util predict() boxes идут в формате cx, cy, w, h
            # и уже нормализованы в диапазон [0..1] для текущего изображения.
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE,
            )

            label_path = label_dir / f"{image_path.stem}.txt"
            yolo_lines = []

            for box, phrase in zip(boxes, phrases):
                class_name = normalize_phrase(str(phrase))
                if class_name is None:
                    continue

                class_id = CLASS_IDS[class_name]

                cx, cy, bw, bh = [float(x) for x in box.tolist()]

                # защита от мусорных значений
                if bw <= 0 or bh <= 0:
                    continue

                # ограничим диапазоны
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 1e-6), 1.0)
                bh = min(max(bh, 1e-6), 1.0)

                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            yolo_lines = deduplicate_lines(yolo_lines)

            with open(label_path, "w", encoding="utf-8") as f:
                for line in yolo_lines:
                    f.write(line + "\n")

            if yolo_lines:
                stats["labeled"] += 1
            else:
                stats["empty"] += 1

            if idx % 50 == 0:
                print(
                    f"{split}: {idx}/{len(image_files)} | "
                    f"размечено={stats['labeled']} пусто={stats['empty']}"
                )

        except Exception as e:
            print(f"Ошибка на {image_path.name}: {e}")
            stats["errors"] += 1

    print(
        f"\n{split}: готово | размечено={stats['labeled']} "
        f"пусто={stats['empty']} ошибок={stats['errors']}"
    )


def main():
    # Проверка наличия весов
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Не найден файл весов: {WEIGHTS_PATH}")

    # Если там случайно текстовый файл "Not Found"
    with open(WEIGHTS_PATH, "rb") as f:
        head = f.read(32)
    if b"Not Found" in head or b"<html" in head.lower():
        raise RuntimeError(
            f"Файл весов {WEIGHTS_PATH} не является моделью. "
            "Внутри текст ошибки/HTML, а не checkpoint."
        )

    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    for split in ["train", "val", "test"]:
        process_split(split, model)

    print("\nАвторазметка завершена.")


if __name__ == "__main__":
    main()