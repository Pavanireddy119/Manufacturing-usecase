"""
Train YOLOv8 for vehicle damage detection
Uses nano model with class weighting and augmentation
"""

import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path

def get_class_weights(dataset_path):
    """Calculate class weights to handle imbalance"""
    classes = []
    classes_file = Path(dataset_path) / "classes.txt"
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    # Count annotations per class
    from collections import defaultdict
    import os
    
    class_counts = defaultdict(int)
    
    labels_dir = Path(dataset_path) / 'labels' / 'train'
    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
    
    # Calculate weights (inverse frequency)
    total = sum(class_counts.values())
    weights = []
    for i in range(len(classes)):
        count = class_counts.get(i, 1)
        weight = total / (len(classes) * count) if count > 0 else 1.0
        weights.append(weight)
    
    # Normalize to sum to number of classes
    weights = np.array(weights)
    weights = weights / weights.mean() * len(classes)
    
    return weights, classes

def train_model():
    """Train YOLOv8n model"""
    
    print("=" * 80)
    print("TRAINING YOLOv8 DAMAGE DETECTION MODEL")
    print("=" * 80)
    
    # Check GPU
    print(f"\nGPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset_path = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\dataset_cleaned"
    
    # Get class weights
    weights, classes = get_class_weights(dataset_path)
    print(f"\nClass weights: {weights}")
    print(f"Classes: {classes}")
    
    # Load model
    print("\n[INFO] Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    print("\n" + "=" * 80)
    print("TRAINING PARAMETERS")
    print("=" * 80)
    
    params = {
        'model': 'yolov8n.pt',
        'data': str(Path(dataset_path) / 'data.yaml'),
        'epochs': 30,
        'imgsz': 416,
        'batch': 8,
        'patience': 5,
        'workers': 2,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs
        'augment': True,
        'mosaic': 1.0,  # Always use mosaic
        'flipud': 0.5,  # 50% probability of vertical flip
        'fliplr': 0.5,  # 50% probability of horizontal flip
        'scale': 0.5,   # Scale augmentation
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation augmentation
        'hsv_v': 0.4,    # HSV-Value augmentation
        'translate': 0.1,  # Image translation
        'perspective': 0.0,
        'degrees': 10,   # Rotation
        'name': 'damage_detection_v1',
        'project': 'runs/detect',
        'exist_ok': False,
        'verbose': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'cfg': None,
        'fraction': 1.0,
        'iou': 0.7,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'copy_paste': 0,
    }
    
    print("\nTraining Configuration:")
    for key, value in params.items():
        print(f"  {key:<20}: {value}")
    
    # Train model
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80 + "\n")
    
    results = model.train(**params)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    # Save best model
    best_model_path = Path("runs/detect/damage_detection_v1/weights/best.pt")
    if best_model_path.exists():
        # Copy to main directory
        import shutil
        output_path = Path("best_damage_model.pt")
        shutil.copy2(best_model_path, output_path)
        print(f"\nBest model saved: {output_path}")
    
    return results

if __name__ == "__main__":
    results = train_model()
