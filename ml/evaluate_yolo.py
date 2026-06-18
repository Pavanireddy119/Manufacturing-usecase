"""
Vehicle Damage Detection - YOLOv8 Evaluation Script
Evaluates trained model on test set and generates comprehensive metrics.
"""

import os
import sys
import yaml
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

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

# Group indices
DAMAGE_GROUP = list(range(0, 6))
LOCATION_GROUP = list(range(6, 15))
SEVERITY_GROUP = list(range(15, 18))

GROUP_NAMES = {
    'damage_type': DAMAGE_GROUP,
    'location': LOCATION_GROUP,
    'severity': SEVERITY_GROUP,
}

METRICS_CSV_HEADER = [
    'class_id', 'class_name', 'precision', 'recall', 'mAP50', 'mAP50-95'
]


def load_model(model_path):
    """Load trained YOLOv8 model."""
    print(f"\nLoading model: {model_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = YOLO(model_path)
    print("Model loaded successfully.")
    
    # Print model info
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Parameters: {n_params:,}")
    
    return model


def evaluate_model(model, data_yaml, output_dir='output/reports'):
    """Run evaluation on test set."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run validation on test set
    print("\nEvaluating on test set...")
    metrics = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.5,
        plots=True,
        save_json=True,
        save_hybrid=True,
    )
    
    # Extract metrics
    box_metrics = metrics.box if hasattr(metrics, 'box') else metrics
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Per-class metrics
    ap50_list = box_metrics.ap50 if hasattr(box_metrics, 'ap50') else []
    ap50_95_list = box_metrics.ap if hasattr(box_metrics, 'ap') else []
    p_list = box_metrics.p if hasattr(box_metrics, 'p') else []
    r_list = box_metrics.r if hasattr(box_metrics, 'r') else []
    
    # Build metrics table
    metrics_data = []
    for i in range(18):
        p = p_list[i] if i < len(p_list) else 0.0
        r = r_list[i] if i < len(r_list) else 0.0
        ap50 = ap50_list[i] if i < len(ap50_list) else 0.0
        ap50_95 = ap50_95_list[i] if i < len(ap50_95_list) else 0.0
        
        metrics_data.append({
            'class_id': i,
            'class_name': CLASS_NAMES[i],
            'precision': p,
            'recall': r,
            'mAP50': ap50,
            'mAP50-95': ap50_95,
            'group': 'damage_type' if i in DAMAGE_GROUP else ('location' if i in LOCATION_GROUP else 'severity'),
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Save metrics CSV
    df_metrics.to_csv(output_dir / 'metrics.csv', index=False)
    print(f"\nMetrics saved to: {output_dir / 'metrics.csv'}")
    
    # Print summary by group
    print(f"\n{'='*60}")
    print("PERFORMANCE BY GROUP")
    print(f"{'='*60}")
    
    for group_name, group_ids in GROUP_NAMES.items():
        group_df = df_metrics[df_metrics['class_id'].isin(group_ids)]
        print(f"\n{group_name.upper()}:")
        print(f"  mAP50:     {group_df['mAP50'].mean():.4f}")
        print(f"  mAP50-95:  {group_df['mAP50-95'].mean():.4f}")
        print(f"  Precision: {group_df['precision'].mean():.4f}")
        print(f"  Recall:    {group_df['recall'].mean():.4f}")
        
        # F1 score = 2 * (P * R) / (P + R)
        avg_p = group_df['precision'].mean()
        avg_r = group_df['recall'].mean()
        f1 = 2 * (avg_p * avg_r) / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        print(f"  F1 Score:  {f1:.4f}")
    
    # Overall metrics
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE")
    print(f"{'='*60}")
    print(f"  mAP50:     {df_metrics['mAP50'].mean():.4f}")
    print(f"  mAP50-95:  {df_metrics['mAP50-95'].mean():.4f}")
    print(f"  Precision: {df_metrics['precision'].mean():.4f}")
    print(f"  Recall:    {df_metrics['recall'].mean():.4f}")
    
    avg_p_all = df_metrics['precision'].mean()
    avg_r_all = df_metrics['recall'].mean()
    f1_all = 2 * (avg_p_all * avg_r_all) / (avg_p_all + avg_r_all) if (avg_p_all + avg_r_all) > 0 else 0
    print(f"  F1 Score:  {f1_all:.4f}")
    
    # Generate performance visualization
    print(f"\nGenerating performance charts...")
    
    # 1. Per-class metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # mAP50
    axes[0, 0].barh(range(18), df_metrics['mAP50'], 
                    color=['#3498db']*6 + ['#2ecc71']*9 + ['#e74c3c']*3)
    axes[0, 0].set_yticks(range(18))
    axes[0, 0].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[0, 0].set_xlabel('mAP50')
    axes[0, 0].set_title('mAP50 by Class')
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    
    # mAP50-95
    axes[0, 1].barh(range(18), df_metrics['mAP50-95'],
                    color=['#3498db']*6 + ['#2ecc71']*9 + ['#e74c3c']*3)
    axes[0, 1].set_yticks(range(18))
    axes[0, 1].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[0, 1].set_xlabel('mAP50-95')
    axes[0, 1].set_title('mAP50-95 by Class')
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    
    # Precision
    axes[1, 0].barh(range(18), df_metrics['precision'],
                    color=['#3498db']*6 + ['#2ecc71']*9 + ['#e74c3c']*3)
    axes[1, 0].set_yticks(range(18))
    axes[1, 0].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_title('Precision by Class')
    axes[1, 0].grid(axis='x', alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    
    # Recall
    axes[1, 1].barh(range(18), df_metrics['recall'],
                    color=['#3498db']*6 + ['#2ecc71']*9 + ['#e74c3c']*3)
    axes[1, 1].set_yticks(range(18))
    axes[1, 1].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_title('Recall by Class')
    axes[1, 1].grid(axis='x', alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    
    plt.suptitle('Model Performance by Class', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=150)
    plt.close()
    
    # 2. Group comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    group_stats = []
    for group_name, group_ids in GROUP_NAMES.items():
        group_df = df_metrics[df_metrics['class_id'].isin(group_ids)]
        avg_p = group_df['precision'].mean()
        avg_r = group_df['recall'].mean()
        avg_ap50 = group_df['mAP50'].mean()
        avg_ap95 = group_df['mAP50-95'].mean()
        group_stats.append({
            'group': group_name,
            'Precision': avg_p,
            'Recall': avg_r,
            'mAP50': avg_ap50,
            'mAP50-95': avg_ap95,
        })
    
    df_group = pd.DataFrame(group_stats)
    df_group.set_index('group').plot(kind='bar', ax=ax, colormap='viridis', width=0.8)
    ax.set_title('Performance by Category Group')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / 'group_performance.png', dpi=150)
    plt.close()
    
    print(f"Performance charts saved to: {output_dir}")
    
    return df_metrics


def test_inference_speed(model, data_yaml, num_runs=100):
    """Measure inference speed."""
    print("\n" + "="*60)
    print("INFERENCE SPEED TEST")
    print("="*60)
    
    import time
    from pathlib import Path
    
    # Load data config
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    base_path = Path(data_cfg.get('path', '.')).resolve()
    test_img_path = base_path / data_cfg.get('test', 'images/test')
    
    if not test_img_path.exists():
        print(f"Test images not found: {test_img_path}")
        return None
    
    # Get test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    test_images = list(test_img_path.iterdir())
    test_images = [f for f in test_images if f.suffix in image_extensions]
    
    if not test_images:
        print("No test images found")
        return None
    
    # Warm-up
    print("Warming up...")
    for _ in range(10):
        _ = model(test_images[0], verbose=False)
    
    # Timing
    print(f"Running {num_runs} inference passes...")
    times = []
    
    for _ in range(num_runs):
        img = random.choice(test_images)
        start = time.time()
        _ = model(img, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    print(f"\nInference Speed Results:")
    print(f"  Mean:    {times.mean()*1000:.1f} ms")
    print(f"  Median:  {np.median(times)*1000:.1f} ms")
    print(f"  Std Dev: {times.std()*1000:.1f} ms")
    print(f"  Min:     {times.min()*1000:.1f} ms")
    print(f"  Max:     {times.max()*1000:.1f} ms")
    print(f"  FPS:     {1.0/times.mean():.1f}")
    
    return {
        'mean_ms': times.mean() * 1000,
        'median_ms': np.median(times) * 1000,
        'std_ms': times.std() * 1000,
        'fps': 1.0 / times.mean(),
    }


def compute_f1_score(metrics_df):
    """Compute F1 score from precision and recall."""
    p = metrics_df['precision'].mean()
    r = metrics_df['recall'].mean()
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0


def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - EVALUATION PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Paths
    model_path = 'output/models/best.pt'
    data_yaml = 'dataset_final/data_yolov8.yaml'
    if not Path(data_yaml).exists():
        data_yaml = 'dataset_final/data.yaml'
    
    output_dir = Path('output/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_yolo.py first or provide the model path.")
        return
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate on test set
    metrics_df = evaluate_model(model, data_yaml, str(output_dir))
    
    # Compute F1 score
    f1 = compute_f1_score(metrics_df)
    print(f"\nOverall F1 Score: {f1:.4f}")
    
    # Test inference speed
    speed_results = test_inference_speed(model, data_yaml, num_runs=50)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir.resolve()}")
    print(f"  - metrics.csv")
    print(f"  - performance_metrics.png")
    print(f"  - group_performance.png")
    print(f"  - confusion_matrix.png (from training)")


if __name__ == '__main__':
    main()