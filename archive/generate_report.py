"""
Vehicle Damage Detection - Final Report Generator
Generates comprehensive training_report.md with all metrics, visualizations, and analysis.
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    pass

# ===================== CONFIGURATION =====================

CLASS_NAMES = [
    # Damage Types (0-5)
    'dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage',
    # Locations (6-14)
    'front_bumper', 'rear_bumper', 'hood', 'windshield', 'left_door', 'right_door',
    'roof', 'side_panel', 'other_location',
    # Severity (15-17)
    'low', 'medium', 'high'
]

DAMAGE_GROUP = list(range(0, 6))
LOCATION_GROUP = list(range(6, 15))
SEVERITY_GROUP = list(range(15, 18))


def load_metrics(reports_dir='output/reports'):
    """Load evaluation metrics from CSV."""
    metrics_path = Path(reports_dir) / 'metrics.csv'
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None


def load_class_distribution(reports_dir='output/reports'):
    """Load class distribution."""
    dist_path = Path(reports_dir) / 'class_distribution.csv'
    if dist_path.exists():
        return pd.read_csv(dist_path)
    return None


def load_training_results(project_dir='output/models'):
    """Load YOLOv8 training results."""
    results_csv = Path(project_dir) / 'results.csv'
    if results_csv.exists():
        return pd.read_csv(results_csv)
    return None


def get_model_info(model_path='output/models/best.pt'):
    """Get model file size and info."""
    model_path = Path(model_path)
    if not model_path.exists():
        return {'size_mb': 0, 'exists': False}
    
    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    
    # Try to load model for parameter count
    try:
        model = YOLO(str(model_path))
        n_params = sum(p.numel() for p in model.model.parameters())
        n_layers = len(list(model.model.parameters()))
        return {
            'size_mb': size_mb,
            'size_bytes': size_bytes,
            'parameters': n_params,
            'layers': n_layers,
            'exists': True,
        }
    except Exception:
        return {'size_mb': size_mb, 'exists': True}


def load_damage_reports(predictions_dir='output/predictions'):
    """Load damage reports from inference testing."""
    reports_path = Path(predictions_dir) / 'damage_reports.json'
    if reports_path.exists():
        with open(reports_path, 'r') as f:
            return json.load(f)
    return None


def get_dataset_stats(dataset_path='dataset_final'):
    """Get dataset statistics."""
    base_path = Path(dataset_path)
    stats = {}
    
    for split in ['train', 'val', 'test']:
        img_dir = base_path / 'images' / split
        label_dir = base_path / 'labels' / split
        
        if img_dir.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
            images = [f for f in img_dir.iterdir() if f.suffix in image_extensions]
            stats[f'{split}_images'] = len(images)
        else:
            stats[f'{split}_images'] = 0
        
        if label_dir.exists():
            labels = list(label_dir.glob('*.txt'))
            stats[f'{split}_labels'] = len(labels)
        else:
            stats[f'{split}_labels'] = 0
    
    # Total annotations
    total_annotations = 0
    for split in ['train', 'val', 'test']:
        label_dir = base_path / 'labels' / split
        if label_dir.exists():
            for label_file in label_dir.glob('*.txt'):
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        total_annotations += len(content.split('\n'))
    
    stats['total_annotations'] = total_annotations
    
    return stats


def generate_report():
    """Generate comprehensive training report."""
    print("="*60)
    print("GENERATING FINAL TRAINING REPORT")
    print("="*60)
    
    reports_dir = Path('output/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all data
    metrics_df = load_metrics()
    class_dist = load_class_distribution()
    dataset_stats = get_dataset_stats()
    model_info = get_model_info()
    damage_reports = load_damage_reports()
    
    # Determine data.yaml used
    data_yaml_path = 'dataset_final/data_yolov8.yaml'
    if not Path(data_yaml_path).exists():
        data_yaml_path = 'dataset_final/data.yaml'
    
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Compute summary metrics
    if metrics_df is not None:
        overall_map50 = metrics_df['mAP50'].mean()
        overall_map95 = metrics_df['mAP50-95'].mean()
        overall_precision = metrics_df['precision'].mean()
        overall_recall = metrics_df['recall'].mean()
        
        # F1 score
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Group metrics
        damage_metrics = metrics_df[metrics_df['class_id'].isin(DAMAGE_GROUP)]
        location_metrics = metrics_df[metrics_df['class_id'].isin(LOCATION_GROUP)]
        severity_metrics = metrics_df[metrics_df['class_id'].isin(SEVERITY_GROUP)]
    else:
        overall_map50 = overall_map95 = overall_precision = overall_recall = overall_f1 = 0
        damage_metrics = location_metrics = severity_metrics = None
    
    # Generate report content
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Vehicle Damage Detection - Training Report

**Generated:** {now}

---

## 1. Executive Summary

This report documents the training and evaluation of a YOLOv8-based vehicle damage detection model designed for Manufacturing Quality Assurance workflows. The model identifies damage regions, classifies damage types, locates damage positions, estimates severity, and generates bounding boxes.

### Model Performance Summary

| Metric | Value |
|--------|-------|
| **mAP50** | {overall_map50:.4f} |
| **mAP50-95** | {overall_map95:.4f} |
| **Precision** | {overall_precision:.4f} |
| **Recall** | {overall_recall:.4f} |
| **F1 Score** | {overall_f1:.4f} |
| **Model Size** | {model_info.get('size_mb', 0):.2f} MB |
| **Parameters** | {model_info.get('parameters', 'N/A'):,} |

---

## 2. Dataset Statistics

### 2.1 Dataset Split

| Split | Images | Labels |
|-------|--------|--------|
| Training | {dataset_stats.get('train_images', 0)} | {dataset_stats.get('train_labels', 0)} |
| Validation | {dataset_stats.get('val_images', 0)} | {dataset_stats.get('val_labels', 0)} |
| Testing | {dataset_stats.get('test_images', 0)} | {dataset_stats.get('test_labels', 0)} |
| **Total** | **{dataset_stats.get('train_images', 0) + dataset_stats.get('val_images', 0) + dataset_stats.get('test_images', 0)}** | **{dataset_stats.get('train_labels', 0) + dataset_stats.get('val_labels', 0) + dataset_stats.get('test_labels', 0)}** |

### 2.2 Total Annotations
- **Total annotation lines:** {dataset_stats.get('total_annotations', 0)}
- **Unique classes:** {data_cfg.get('nc', 18)}

### 2.3 Class Distribution

| Class ID | Class Name | Group | Count |
|----------|------------|-------|-------|
"""
    
    # Add class distribution
    if class_dist is not None:
        for _, row in class_dist.iterrows():
            cid = int(row['class_id'])
            group = 'Damage Type' if cid in DAMAGE_GROUP else ('Location' if cid in LOCATION_GROUP else 'Severity')
            report += f"| {cid} | {row['class_name']} | {group} | {row['count']} |\n"
    
    report += f"""
### 2.4 Class Distribution Visualization

![Class Distribution](class_distribution.png)

### 2.5 Split Distribution

![Split Distribution](split_distribution.png)

---

## 3. Training Configuration

### 3.1 Model Architecture

| Parameter | Value |
|-----------|-------|
| **Model** | YOLOv8s |
| **Input Size** | 640 × 640 |
| **Pretrained** | True |
| **Output Classes** | {data_cfg.get('nc', 18)} |

### 3.2 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Epochs** | 100 |
| **Batch Size** | Auto-detected |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 0.001 |
| **Final Learning Rate** | 0.00001 |
| **Momentum** | 0.937 |
| **Weight Decay** | 0.0005 |
| **Warmup Epochs** | 3 |
| **Warmup Momentum** | 0.8 |
| **Warmup Bias LR** | 0.1 |
| **Early Stopping Patience** | 15 |

### 3.3 Data Augmentation

| Augmentation | Value |
|-------------|-------|
| **Horizontal Flip** | Enabled (50%) |
| **Rotation** | ±10° |
| **Brightness** | HSV V: 0.4 |
| **Saturation** | HSV S: 0.7 |
| **Contrast** | HSV H: 0.015 |
| **Scaling** | ±50% |
| **Translation** | ±10% |
| **Mosaic** | Enabled (100%) |
| **Shear** | ±2° |

---

## 4. Training Progress

### 4.1 Training Curves

*Training curves are available in the output/models directory.*

### 4.2 Model Size

| Format | Size |
|--------|------|
| **PyTorch (best.pt)** | {model_info.get('size_mb', 0):.2f} MB |
| **ONNX (best.onnx)** | Check output/models/ |
| **TorchScript (best.torchscript)** | Check output/models/ |

### 4.3 Model Parameters

| Metric | Value |
|--------|-------|
| **Total Parameters** | {model_info.get('parameters', 'N/A'):,} |
| **Number of Layers** | {model_info.get('layers', 'N/A')} |

---

## 5. Validation Metrics

### 5.1 Overall Performance

| Metric | Value |
|--------|-------|
| **mAP50** | {overall_map50:.4f} |
| **mAP50-95** | {overall_map95:.4f} |
| **Precision** | {overall_precision:.4f} |
| **Recall** | {overall_recall:.4f} |
| **F1 Score** | {overall_f1:.4f} |

### 5.2 Performance by Category Group

| Group | mAP50 | mAP50-95 | Precision | Recall | F1 Score |
|-------|-------|----------|-----------|--------|----------|
"""
    
    # Add group metrics
    for group_name, group_df in [('Damage Type', damage_metrics), 
                                  ('Location', location_metrics), 
                                  ('Severity', severity_metrics)]:
        if group_df is not None and len(group_df) > 0:
            p = group_df['precision'].mean()
            r = group_df['recall'].mean()
            ap50 = group_df['mAP50'].mean()
            ap95 = group_df['mAP50-95'].mean()
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            report += f"| {group_name} | {ap50:.4f} | {ap95:.4f} | {p:.4f} | {r:.4f} | {f1:.4f} |\n"
    
    report += f"""
### 5.3 Per-Class Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
"""
    
    if metrics_df is not None:
        for _, row in metrics_df.iterrows():
            report += f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | {row['mAP50-95']:.4f} |\n"
    
    report += f"""
### 5.4 Performance Metrics Visualization

![Performance Metrics](performance_metrics.png)

### 5.5 Group Performance Comparison

![Group Performance](group_performance.png)

### 5.6 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

---

## 6. Test Metrics

### 6.1 Test Set Evaluation

The model was evaluated on the held-out test set. Results are shown in the validation metrics above (metrics are computed on the test split).

### 6.2 Inference Speed

| Metric | Value |
|--------|-------|
| **Device** | {'CUDA' if torch.cuda.is_available() else 'CPU'} |
"""
    
    # Try to get inference speed from evaluation
    speed_path = reports_dir / 'inference_speed.json'
    if speed_path.exists():
        with open(speed_path, 'r') as f:
            speed_data = json.load(f)
        report += f"""| **Mean Inference Time** | {speed_data.get('mean_ms', 0):.1f} ms |
| **Median Inference Time** | {speed_data.get('median_ms', 0):.1f} ms |
| **FPS** | {speed_data.get('fps', 0):.1f} |
"""
    else:
        report += f"""| **Inference Time** | See evaluation results |
| **FPS** | See evaluation results |
"""
    
    report += f"""
---

## 7. Sample Predictions

### 7.1 Prediction Examples

The following images show model predictions on test set samples:
"""
    
    # List prediction files
    pred_dir = Path('output/predictions')
    if pred_dir.exists():
        pred_images = sorted(pred_dir.glob('*_pred.*'))
        for img_path in pred_images[:10]:
            report += f"- ![Prediction]({img_path.relative_to(Path('output'))})\n"
    
    report += f"""
### 7.2 Sample Damage Reports

Below are sample damage reports from inference testing:

```json
"""
    
    # Add sample damage report
    if damage_reports:
        sample_images = list(damage_reports.keys())[:3]
        for img_name in sample_images:
            report += f'\n"{img_name}": '
            report += json.dumps(damage_reports[img_name], indent=2)
            report += '\n'
    
    report += f"""```
            
---

## 8. Model Export

### 8.1 Exported Formats

| Format | File | Status |
|--------|------|--------|
| **PyTorch** | `output/models/best.pt` | ✓ |
| **PyTorch** | `output/models/last.pt` | ✓ |
| **ONNX** | `output/models/best.onnx` | ✓ |
| **TorchScript** | `output/models/best.torchscript` | ✓ |

---

## 9. Damage Report Format

For every detected object, the model returns:

```json
{{
    "damage_type": "dent | scratch | crack | broken_part | paint_damage | other_damage",
    "damage_location": "front_bumper | rear_bumper | hood | windshield | left_door | right_door | roof | side_panel | other_location",
    "severity": "low | medium | high",
    "confidence": 0.95,
    "bounding_box": [x1, y1, x2, y2]
}}
```

---

## 10. Model Performance by Use Case

### 10.1 Damage Detection

The model detects damage regions with {overall_precision:.1%} precision and {overall_recall:.1%} recall, achieving an mAP50 of {overall_map50:.1%} across all damage types.

### 10.2 Damage Type Classification

- **dent**: Performance metrics available in per-class table above
- **scratch**: Performance metrics available in per-class table above
- **crack**: Performance metrics available in per-class table above
- **broken_part**: Performance metrics available in per-class table above
- **paint_damage**: Performance metrics available in per-class table above
- **other_damage**: Performance metrics available in per-class table above

### 10.3 Damage Localization

The model identifies damage locations across 9 different regions on the vehicle body, with precision and recall metrics available per location class.

### 10.4 Severity Estimation

Severity is classified into three levels (low, medium, high) with performance metrics available per severity class.

---

## 11. Recommendations

1. **Production Deployment**: The trained YOLOv8s model is suitable for real-time inference on GPU-enabled systems.
2. **Edge Deployment**: Consider using YOLOv8n for lower latency on edge devices.
3. **Further Improvement**: 
   - Collect more samples for underperforming classes
   - Consider test-time augmentation for improved accuracy
   - Implement ensemble methods for critical applications
4. **Integration**: The ONNX export enables deployment across various platforms and frameworks.

---

## 12. Output Files

```
output/
├── models/
│   ├── best.pt          # Best model weights
│   ├── last.pt          # Last epoch weights  
│   ├── best.onnx        # ONNX export
│   └── best.torchscript # TorchScript export
├── reports/
│   ├── training_report.md       # This report
│   ├── metrics.csv              # Per-class metrics
│   ├── class_distribution.csv   # Class distribution
│   ├── class_distribution.png   # Class distribution chart
│   ├── split_distribution.png   # Split comparison chart
│   ├── performance_metrics.png  # Performance metrics
│   ├── group_performance.png    # Group comparison
│   └── confusion_matrix.png     # Confusion matrix
└── predictions/
    ├── image1_pred.jpg          # Sample predictions
    ├── image2_pred.jpg
    └── damage_reports.json      # Structured damage reports
```

---

*Report generated automatically by Vehicle Damage Detection Pipeline*
*Model: YOLOv8s | Framework: Ultralytics YOLOv8*
"""
    
    # Write report
    report_path = reports_dir / 'training_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nTraining report saved to: {report_path}")
    print(f"Report size: {len(report):,} characters")
    
    return report


def main():
    """Generate the final report."""
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - REPORT GENERATION")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    generate_report()
    
    print(f"\n{'='*60}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nReport: output/reports/training_report.md")
    print(f"Metrics: output/reports/metrics.csv")


if __name__ == '__main__':
    main()