"""Train a binary quality classifier (damage vs whole) with YOLOv8-cls.

Uses the balanced classification dataset in Dataset/data1a
(920+920 train, 230+230 val). Rebuilds the Ultralytics-style train/val
folder layout if needed, trains a small pretrained classifier via transfer
learning, and copies the best weights into the integration package.

Run from the ml/ directory:

    ../backend/.venv/Scripts/python.exe train_quality_classifier.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ML_DIR = Path(__file__).resolve().parent
SRC = ML_DIR / "Dataset" / "data1a"
CLASSIFY = ML_DIR / "classify_dataset"
DEST_WEIGHTS = ML_DIR / "integration_package" / "quality_classifier.pt"

# Map source folders -> (split, class name)
LAYOUT = {
    ("training", "00-damage"): ("train", "damage"),
    ("training", "01-whole"): ("train", "whole"),
    ("validation", "00-damage"): ("val", "damage"),
    ("validation", "01-whole"): ("val", "whole"),
}


def build_dataset() -> None:
    if CLASSIFY.exists():
        return
    for (split_src, cls_src), (split, cls) in LAYOUT.items():
        src = SRC / split_src / cls_src
        dst = CLASSIFY / split / cls
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    print("Built classification dataset at", CLASSIFY)


def main() -> None:
    build_dataset()

    model = YOLO("yolov8n-cls.pt")  # pretrained ImageNet classifier
    results = model.train(
        data=str(CLASSIFY),
        epochs=20,
        imgsz=160,
        batch=16,
        patience=5,
        project=str(ML_DIR / "runs_classify"),
        name="quality",
        exist_ok=True,
        verbose=True,
        plots=False,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    DEST_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, DEST_WEIGHTS)
    print(f"\nBest weights copied to {DEST_WEIGHTS}")
    print(f"Top-1 val accuracy: {getattr(results, 'top1', 'see results dir')}")


if __name__ == "__main__":
    main()
