"""
Train YOLOv8s on the fixed dataset (datasets/dataset_yolo_fixed/).
- Only 6 damage type classes
- No triplicate annotations
- epochs=100, imgsz=640, batch=auto, patience=30
"""
import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

# Configuration
DATA_YAML = 'datasets/dataset_yolo_fixed/data.yaml'
MODEL_NAME = 'yolov8s'
EPOCHS = 100
IMGSZ = 640
BATCH = -1  # auto
PATIENCE = 30
PROJECT = 'output'
NAME = 'yolo_fixed'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    print("=" * 60)
    print("YOLOv8s TRAINING ON FIXED DATASET")
    print("=" * 60)
    print(f"Data: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMGSZ}")
    print(f"Batch: auto")
    print(f"Patience: {PATIENCE}")
    print(f"Device: {DEVICE}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data.yaml
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    print(f"\nClasses ({data_cfg['nc']}): {data_cfg['names']}")
    
    set_seed(SEED)
    
    # Initialize model from pretrained
    print(f"\nLoading pretrained {MODEL_NAME}...")
    model = YOLO(f'{MODEL_NAME}.pt')
    
    # Print model info
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    print(f"\nStarting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        device=DEVICE,
        workers=4,
        seed=SEED,
        plots=True,
        val=True,
        save=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    print(f"\nTraining complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Copy results to output directories
    import shutil
    train_dir = Path(PROJECT) / NAME
    
    # Create output directories
    os.makedirs('outputs/model_outputs/models_fixed', exist_ok=True)
    os.makedirs('outputs/model_outputs/reports_fixed', exist_ok=True)
    os.makedirs('outputs/predictions/predictions_fixed', exist_ok=True)
    
    # Copy weights
    for w in ['best.pt', 'last.pt']:
        src = train_dir / 'weights' / w
        if src.exists():
            shutil.copy(src, f'outputs/model_outputs/models_fixed/{w}')
            print(f"  Copied {w}")
    
    # Copy training plots
    for fname in ['confusion_matrix.png', 'PR_curve.png', 'P_curve.png', 'R_curve.png', 
                  'F1_curve.png', 'results.csv', 'results.png', 'labels.jpg', 'labels_correlogram.jpg']:
        src = train_dir / fname
        if src.exists():
            shutil.copy(src, f'outputs/model_outputs/reports_fixed/{fname}')
            print(f"  Copied {fname}")
    
    print(f"\nOutput files:")
    print(f"  - Models: outputs/model_outputs/models_fixed/")
    print(f"  - Reports: outputs/model_outputs/reports_fixed/")
    print(f"  - Predictions: outputs/predictions/predictions_fixed/")

if __name__ == '__main__':
    main()
