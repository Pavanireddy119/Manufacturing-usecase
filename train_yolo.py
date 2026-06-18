"""
Vehicle Damage Detection - YOLOv8 Training Script
Trains a YOLOv8 model for damage detection, type classification,
location identification, and severity estimation.
"""

import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Install ultralytics if not available
try:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    from ultralytics.utils import LOGGER
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device
    from ultralytics.utils import LOGGER

# ===================== CONFIGURATION =====================

CONFIG = {
    # Dataset
    'data_yaml': 'dataset_final/data.yaml',
    'dataset_path': 'dataset_final',
    
    # Model selection: 'yolov8n' for testing, 'yolov8s' for production
    'model_name': 'yolov8s',
    
    # Training parameters
    'img_size': 640,
    'batch_size': -1,  # -1 for auto-batch
    'epochs': 100,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Early stopping
    'patience': 15,
    'save_period': -1,  # -1 to save only best and last
    
    # Data augmentation
    'hsv_h': 0.015,     # Hue
    'hsv_s': 0.7,       # Saturation
    'hsv_v': 0.4,       # Value (brightness)
    'degrees': 10.0,    # Rotation
    'translate': 0.1,   # Translation
    'scale': 0.5,       # Scaling
    'shear': 2.0,       # Shear
    'perspective': 0.0, # Perspective
    'flipud': 0.0,      # Flip up-down
    'fliplr': 0.5,      # Flip left-right
    'mosaic': 1.0,      # Mosaic augmentation
    'mixup': 0.0,       # Mixup
    'copy_paste': 0.0,  # Copy-paste
    
    # Output
    'project': 'output',
    'name': 'models',
    'exist_ok': True,
    'pretrained': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Workers
    'workers': 4,
    
    # Seeds
    'seed': 42,
}

CLASS_NAMES = [
    # Damage Types (0-5)
    'dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage',
    # Locations (6-14)
    'front_bumper', 'rear_bumper', 'hood', 'windshield', 'left_door', 'right_door',
    'roof', 'side_panel', 'other_location',
    # Severity (15-17)
    'low', 'medium', 'high'
]

# Class groupings
DAMAGE_TYPES = list(range(0, 6))
DAMAGE_LOCATIONS = list(range(6, 15))
SEVERITY_LEVELS = list(range(15, 18))


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def validate_dataset(data_yaml_path):
    """Validate dataset structure and annotations."""
    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)
    
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    dataset_path = Path(data_cfg.get('path', ''))
    
    splits = {
        'train': Path(data_cfg['train']) if isinstance(data_cfg['train'], str) else Path(data_cfg['train'][0]),
        'val': Path(data_cfg['val']) if isinstance(data_cfg['val'], str) else Path(data_cfg['val'][0]),
        'test': Path(data_cfg['test']) if isinstance(data_cfg['test'], str) else Path(data_cfg['test'][0]),
    }
    
    # Make paths absolute if relative
    base_path = Path(data_cfg.get('path', '.')).resolve()
    for split in splits:
        if not splits[split].is_absolute():
            splits[split] = base_path / splits[split]
    
    print(f"\nDataset path: {base_path}")
    print(f"Number of classes: {data_cfg['nc']}")
    print(f"Class names: {data_cfg['names']}")
    
    validation_results = {}
    
    for split_name, split_path in splits.items():
        img_path = split_path
        label_path = base_path / 'labels' / split_name
        
        print(f"\n--- {split_name.upper()} Split ---")
        print(f"Images path: {img_path}")
        print(f"Labels path: {label_path}")
        
        if not img_path.exists():
            print(f"  WARNING: Images directory not found: {img_path}")
            validation_results[split_name] = {'valid': False, 'error': 'Images directory not found'}
            continue
        
        if not label_path.exists():
            print(f"  WARNING: Labels directory not found: {label_path}")
            validation_results[split_name] = {'valid': False, 'error': 'Labels directory not found'}
            continue
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = {f.stem: f for f in img_path.iterdir() if f.suffix in image_extensions}
        label_files = {f.stem: f for f in label_path.iterdir() if f.suffix == '.txt'}
        
        print(f"  Images found: {len(image_files)}")
        print(f"  Labels found: {len(label_files)}")
        
        # Check image-label matching
        images_with_labels = set(image_files.keys()) & set(label_files.keys())
        images_without_labels = set(image_files.keys()) - set(label_files.keys())
        labels_without_images = set(label_files.keys()) - set(image_files.keys())
        
        print(f"  Images with labels: {len(images_with_labels)}")
        if images_without_labels:
            print(f"  WARNING: {len(images_without_labels)} images without labels")
        if labels_without_images:
            print(f"  WARNING: {len(labels_without_images)} labels without images")
        
        # Validate annotation format
        total_boxes = 0
        total_damage_groups = 0
        class_counts = Counter()
        invalid_labels = []
        
        for img_stem in list(images_with_labels)[:5]:  # Check first 5
            label_file = label_path / f"{img_stem}.txt"
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            invalid_labels.append((img_stem, f"Incorrect format: {len(parts)} values"))
                        else:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= data_cfg['nc']:
                                invalid_labels.append((img_stem, f"Invalid class ID: {class_id}"))
                            # Validate coordinates
                            _, x, y, w, h = map(float, parts)
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                invalid_labels.append((img_stem, f"Coordinates out of range: {x},{y},{w},{h}"))
        
        if invalid_labels:
            print(f"  WARNING: Found {len(invalid_labels)} invalid annotations:")
            for stem, err in invalid_labels[:3]:
                print(f"    - {stem}: {err}")
        else:
            print(f"  Annotation format: VALID")
        
        # Count total annotations
        for img_stem in images_with_labels:
            label_file = label_path / f"{img_stem}.txt"
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_boxes += 1
                    # Count groups of 3 (damage_type, location, severity)
                    total_damage_groups += len(lines) // 3
        
        validation_results[split_name] = {
            'valid': True,
            'images': len(images_with_labels),
            'total_boxes': total_boxes,
            'damage_groups': total_damage_groups,
            'class_counts': class_counts,
        }
        
        print(f"  Total annotations (lines): {total_boxes}")
        print(f"  Damage instances: {total_damage_groups}")
    
    print("\n" + "="*60)
    print("DATASET VALIDATION COMPLETE")
    print("="*60)
    
    return validation_results


def generate_dataset_statistics(data_yaml_path, output_dir):
    """Generate dataset statistics and visualizations."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    base_path = Path(data_cfg.get('path', '.')).resolve()
    reports_dir = Path(output_dir) / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    all_class_counts = Counter()
    split_counts = {}
    split_images = {}
    
    for split_name in splits:
        img_path = base_path / 'images' / split_name
        label_path = base_path / 'labels' / split_name
        
        if not img_path.exists() or not label_path.exists():
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = {f.stem: f for f in img_path.iterdir() if f.suffix in image_extensions}
        label_files = {f.stem: f for f in label_path.iterdir() if f.suffix == '.txt'}
        
        images_with_labels = set(image_files.keys()) & set(label_files.keys())
        split_images[split_name] = len(images_with_labels)
        
        class_counts = Counter()
        for img_stem in images_with_labels:
            label_file = label_path / f"{img_stem}.txt"
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_counts[int(parts[0])] += 1
                            all_class_counts[int(parts[0])] += 1
        
        split_counts[split_name] = class_counts
    
    # Print statistics
    print(f"\nSplit distribution:")
    for split_name in splits:
        if split_name in split_images:
            print(f"  {split_name}: {split_images[split_name]} images")
    
    print(f"\nClass distribution (all splits):")
    class_data = []
    for class_id in range(data_cfg['nc']):
        count = all_class_counts.get(class_id, 0)
        class_name = data_cfg['names'][class_id]
        print(f"  {class_name} (ID {class_id}): {count}")
        class_data.append({'class_id': class_id, 'class_name': class_name, 'count': count})
    
    # Save class distribution CSV
    df = pd.DataFrame(class_data)
    df.to_csv(reports_dir / 'class_distribution.csv', index=False)
    
    # Plot class distribution
    plt.figure(figsize=(14, 8))
    colors = ['#3498db'] * 6 + ['#2ecc71'] * 9 + ['#e74c3c'] * 3
    bars = plt.bar(range(data_cfg['nc']), [all_class_counts.get(i, 0) for i in range(data_cfg['nc'])], color=colors)
    plt.xticks(range(data_cfg['nc']), data_cfg['names'], rotation=45, ha='right', fontsize=9)
    plt.ylabel('Count')
    plt.title('Class Distribution Across Dataset')
    plt.tight_layout()
    
    # Add legend for groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Damage Types (0-5)'),
        Patch(facecolor='#2ecc71', label='Locations (6-14)'),
        Patch(facecolor='#e74c3c', label='Severity (15-17)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(reports_dir / 'class_distribution.png', dpi=150)
    plt.close()
    
    # Plot split comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, split_name in enumerate(splits):
        if split_name in split_counts:
            counts = [split_counts[split_name].get(j, 0) for j in range(data_cfg['nc'])]
            axes[i].bar(range(data_cfg['nc']), counts, color='#3498db')
            axes[i].set_title(f'{split_name.upper()} Split')
            axes[i].set_xticks(range(data_cfg['nc']))
            axes[i].set_xticklabels(data_cfg['names'], rotation=45, ha='right', fontsize=7)
            axes[i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(reports_dir / 'split_distribution.png', dpi=150)
    plt.close()
    
    # Box/annotation statistics
    print(f"\nAnnotation Statistics:")
    all_boxes = []
    for split_name in splits:
        if split_name not in split_counts:
            continue
        label_path = base_path / 'labels' / split_name
        if not label_path.exists():
            continue
        
        for label_file in label_path.iterdir():
            if label_file.suffix == '.txt':
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        for line in content.split('\n'):
                            parts = line.strip().split()
                            if len(parts) == 5:
                                _, x, y, w, h = map(float, parts)
                                all_boxes.append({'w': w, 'h': h, 'area': w * h})
    
    if all_boxes:
        df_boxes = pd.DataFrame(all_boxes)
        print(f"  Total bounding boxes: {len(df_boxes)}")
        print(f"  Average box width: {df_boxes['w'].mean():.4f}")
        print(f"  Average box height: {df_boxes['h'].mean():.4f}")
        print(f"  Average box area: {df_boxes['area'].mean():.6f}")
        
        # Plot box size distribution
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(df_boxes['w'], bins=50, color='#3498db', alpha=0.7)
        axes[0].set_xlabel('Width (normalized)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Box Width Distribution')
        
        axes[1].hist(df_boxes['h'], bins=50, color='#2ecc71', alpha=0.7)
        axes[1].set_xlabel('Height (normalized)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Box Height Distribution')
        
        axes[2].hist(df_boxes['area'], bins=50, color='#e74c3c', alpha=0.7)
        axes[2].set_xlabel('Area (normalized)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Box Area Distribution')
        
        plt.tight_layout()
        plt.savefig(reports_dir / 'box_size_distribution.png', dpi=150)
        plt.close()
    
    return {
        'split_images': split_images,
        'class_distribution': all_class_counts,
        'total_annotations': sum(all_class_counts.values()),
    }


def prepare_yolo_data(data_yaml_path):
    """Prepare a modified data.yaml for standard YOLO training.
    The original annotation format has 3 lines per damage (type, location, severity).
    We'll use the original format directly since YOLOv8 handles multi-class per box.
    """
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Ensure paths are correct
    base_path = Path(data_cfg.get('path', '.')).resolve()
    
    # Create a working copy of data.yaml
    working_yaml = {
        'path': str(base_path),
        'train': str(base_path / 'images' / 'train'),
        'val': str(base_path / 'images' / 'val'),
        'test': str(base_path / 'images' / 'test'),
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
    }
    
    working_yaml_path = Path('dataset_final/data_yolov8.yaml')
    with open(working_yaml_path, 'w') as f:
        yaml.dump(working_yaml, f, default_flow_style=False)
    
    print(f"Prepared YOLOv8 data config: {working_yaml_path}")
    return working_yaml_path


def train_yolov8(config):
    """Train YOLOv8 model."""
    print("\n" + "="*60)
    print("YOLOv8 TRAINING")
    print("="*60)
    
    set_seed(config['seed'])
    
    # Prepare data config
    data_yaml = prepare_yolo_data(config['data_yaml'])
    
    # Select device
    device = select_device(config['device'])
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize model
    print(f"\nLoading model: {config['model_name']}")
    model = YOLO(f"{config['model_name']}.pt") if config['pretrained'] else YOLO(f"{config['model_name']}.yaml")
    
    # Print model info
    print(f"Model: {config['model_name']}")
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': config['epochs'],
        'imgsz': config['img_size'],
        'batch': config['batch_size'],
        'optimizer': config['optimizer'],
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': config['warmup_momentum'],
        'warmup_bias_lr': config['warmup_bias_lr'],
        'patience': config['patience'],
        'save_period': config['save_period'],
        'project': config['project'],
        'name': config['name'],
        'exist_ok': config['exist_ok'],
        'pretrained': config['pretrained'],
        'device': config['device'],
        'workers': config['workers'],
        'seed': config['seed'],
        'plots': True,
        'val': True,
        
        # Data augmentation
        'hsv_h': config['hsv_h'],
        'hsv_s': config['hsv_s'],
        'hsv_v': config['hsv_v'],
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'mixup': config['mixup'],
        'copy_paste': config['copy_paste'],
        
    }
    
    print(f"\nTraining Configuration:")
    print(f"  Image Size: {train_args['imgsz']}")
    print(f"  Batch Size: {train_args['batch']}")
    print(f"  Epochs: {train_args['epochs']}")
    print(f"  Optimizer: {train_args['optimizer']}")
    print(f"  Learning Rate: {train_args['lr0']}")
    print(f"  Early Stopping Patience: {train_args['patience']}")
    print(f"  Augmentation: HSV({config['hsv_h']},{config['hsv_s']},{config['hsv_v']}), "
          f"Rot({config['degrees']}), FlipLR({config['fliplr']}), Mosaic({config['mosaic']})")
    
    print("\nStarting training...")
    results = model.train(**train_args)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return results


def main():
    """Main training pipeline."""
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - YOLOv8 TRAINING PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 1. Validate dataset
    print("\n[Step 1/5] Validating dataset...")
    validation_results = validate_dataset(CONFIG['data_yaml'])
    
    # 2. Generate dataset statistics
    print("\n[Step 2/5] Generating dataset statistics...")
    stats = generate_dataset_statistics(CONFIG['data_yaml'], CONFIG['project'])
    
    # 3. Train model
    print("\n[Step 3/5] Training YOLOv8 model...")
    train_results = train_yolov8(CONFIG)
    
    # 4. Copy best model to output
    print("\n[Step 4/5] Copying model files...")
    import shutil
    models_dir = Path(CONFIG['project']) / CONFIG['name']
    output_models = Path('output') / 'models'
    output_models.mkdir(parents=True, exist_ok=True)
    
    # Copy best.pt and last.pt
    best_src = models_dir / 'weights' / 'best.pt'
    last_src = models_dir / 'weights' / 'last.pt'
    
    if best_src.exists():
        shutil.copy(best_src, output_models / 'best.pt')
        print(f"  Copied best.pt to {output_models}")
    if last_src.exists():
        shutil.copy(last_src, output_models / 'last.pt')
        print(f"  Copied last.pt to {output_models}")
    
    # 5. Export model
    print("\n[Step 5/5] Exporting model formats...")
    try:
        model = YOLO(str(best_src) if best_src.exists() else str(last_src))
        
        # Export to ONNX
        model.export(format='onnx', imgsz=CONFIG['img_size'])
        onnx_file = models_dir / 'weights' / 'best.onnx'
        if onnx_file.exists():
            shutil.copy(onnx_file, output_models / 'best.onnx')
            print(f"  Exported best.onnx to {output_models}")
        
        # Export to TorchScript
        model.export(format='torchscript', imgsz=CONFIG['img_size'])
        ts_file = models_dir / 'weights' / 'best.torchscript'
        if ts_file.exists():
            shutil.copy(ts_file, output_models / 'best.torchscript')
            print(f"  Exported best.torchscript to {output_models}")
    except Exception as e:
        print(f"  Export error: {e}")
    
    # Copy reports
    shutil.copytree(models_dir / 'confusion_matrix.png', Path('output') / 'reports' / 'confusion_matrix.png', 
                    dirs_exist_ok=True) if (models_dir / 'confusion_matrix.png').exists() else None
    shutil.copytree(models_dir / 'results.csv', Path('output') / 'reports' / 'metrics.csv',
                    dirs_exist_ok=True) if (models_dir / 'results.csv').exists() else None
    
    print(f"\n{'='*60}")
    print(f"TRAINING PIPELINE COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - Models: output/models/best.pt, output/models/last.pt")
    print(f"  - Exported: output/models/best.onnx, output/models/best.torchscript")
    print(f"  - Reports: output/reports/")


if __name__ == '__main__':
    main()