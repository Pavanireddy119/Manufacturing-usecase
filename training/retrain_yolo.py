"""
YOLOv8 Retraining Script - Clean Dataset
Uses restructured dataset with only damage type classes (6 classes)
"""
import os, sys, shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# Config
DATA_YAML = os.path.abspath('datasets/dataset_yolo_restructured/damage_only.yaml')
MODEL_NAME = 'yolov8s'
EPOCHS = 100
IMG_SIZE = 640
BATCH = -1  # auto-batch
PATIENCE = 20
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

print("=" * 60)
print("YOLOv8 RETRAINING - DAMAGE TYPE DETECTION")
print("=" * 60)
print(f"Data: {DATA_YAML}")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}")
print(f"Device: {DEVICE}")

# Validate data
if not os.path.exists(DATA_YAML):
    print(f"ERROR: Data config not found: {DATA_YAML}")
    sys.exit(1)

# Load model
print("\nLoading YOLOv8s pretrained model...")
model = YOLO(f'{MODEL_NAME}.pt')

# Train
print("\nStarting training...")
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    patience=PATIENCE,
    device=DEVICE,
    project='output',
    name='retrained',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    # Other
    workers=4,
    seed=42,
    plots=True,
    val=True,
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

# Copy best model
src = Path('outputs/model_outputs/retrained/weights/best.pt')
dst = Path('outputs/model_outputs/models/yolov8s_damage.pt')
dst.parent.mkdir(exist_ok=True)
shutil.copy(src, dst)
print(f"Model saved to: {dst}")

# Show results
print("\nTraining results saved to:")
print(f"  - outputs/model_outputs/retrained/")
print(f"  - outputs/model_outputs/models/yolov8s_damage.pt")
