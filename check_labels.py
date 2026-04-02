import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset-vehicles/detect")
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

def load_image(image_path):
    return cv2.imread(str(image_path))

def load_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

def plot_image_with_bboxes(image_path, labels):
    # Load image
    image = load_image(image_path)
    
    for label in labels:
        # Get bbox parameters
        class_id, cx, cy, w, h = label
        h, w, _ = image.shape
        x1 = int((cx - w / 2) * w)
        y1 = int((cy - h / 2) * h)
        x2 = int((cx + w / 2) * w)
        y2 = int((cy + h / 2) * h)
        
        # Draw the bounding box
        color = (0, 255, 0)  # Green for bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Convert BGR (OpenCV) to RGB (for plotting)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def check_image(image_name, label_dir, image_dir):
    label_path = label_dir / f"{image_name}.txt"
    image_path = image_dir / f"{image_name}.jpg"

    if label_path.exists() and image_path.exists():
        labels = load_labels(label_path)
        plot_image_with_bboxes(image_path, labels)
    else:
        print(f"Missing label or image for {image_name}")

def check_dataset():
    image_dir = DATASET_DIR / "images" / "train"
    label_dir = DATASET_DIR / "labels" / "train"
    
    # Check 20 random images
    for i, image_file in enumerate(image_dir.glob("*.jpg")):
        image_name = image_file.stem
        check_image(image_name, label_dir, image_dir)
        if i >= 19:  # limit to 20 images for checking
            break

if __name__ == "__main__":
    check_dataset()