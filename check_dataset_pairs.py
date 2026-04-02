from pathlib import Path

DATASET_DIR = Path("dataset-vehicles/detect")
EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

def check_split(split: str):
    image_dir = DATASET_DIR / "images" / split
    label_dir = DATASET_DIR / "labels" / split

    missing_images = []
    missing_labels = []

    image_stems = {p.stem for p in image_dir.iterdir() if p.is_file()}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}

    for stem in sorted(label_stems - image_stems):
        missing_images.append(stem)

    for stem in sorted(image_stems - label_stems):
        missing_labels.append(stem)

    print(f"\n[{split}]")
    print(f"images: {len(image_stems)}")
    print(f"labels: {len(label_stems)}")
    print(f"labels without images: {len(missing_images)}")
    print(f"images without labels: {len(missing_labels)}")

    if missing_images[:10]:
        print("examples missing images:", missing_images[:10])
    if missing_labels[:10]:
        print("examples missing labels:", missing_labels[:10])

def main():
    for split in ["train", "val", "test"]:
        check_split(split)

if __name__ == "__main__":
    main()