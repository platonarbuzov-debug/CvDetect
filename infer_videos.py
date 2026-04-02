import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1 - 8, 0)
        cv2.rectangle(frame, (x1, max(y_text - th - 6, 0)), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(frame, label, (x1 + 3, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def infer_video(model, device, video_path: Path, out_path: Path, imgsz=640, conf=0.25, iou=0.45):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    names = model.names if hasattr(model, "names") else None
    colors = {}

    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    imgsz = int(np.ceil(imgsz / stride) * stride)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img = letterbox(frame, new_shape=imgsz, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        with torch.no_grad():
            pred = model(im)[0]
            pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, classes=None, agnostic=False)

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, confv, cls in det.tolist():
                cls = int(cls)
                if cls not in colors:
                    colors[cls] = tuple(int(x) for x in np.random.randint(0, 255, size=3))

                cname = str(cls) if names is None else names[cls]
                label = f"{cname} {confv:.2f}"
                draw_box(frame, xyxy, label, colors[cls])

        out.write(frame)

    out.release()
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt")
    parser.add_argument("--source", type=str, default="video_fly")
    parser.add_argument("--out", type=str, default="video_out")
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    model = attempt_load(args.weights, map_location=device)
    model.eval()

    videos = sorted([p for p in source_dir.glob("*.mp4") if p.is_file()])
    if not videos:
        raise SystemExit(f"No .mp4 videos found in: {source_dir.resolve()}")

    for vp in videos:
        out_path = out_dir / f"{vp.stem}_boxed.mp4"
        print(f"[INFO] {vp.name} -> {out_path.name}")
        infer_video(model, device, vp, out_path, imgsz=args.img, conf=args.conf, iou=args.iou)

    print(f"[DONE] Saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
