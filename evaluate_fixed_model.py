"""
Evaluate the retrained YOLOv8 model on the fixed dataset.
Generates:
- confusion_matrix.png (copy)
- PR_curve.png (copy)
- results.csv (metrics)
- mAP report
- retraining_report.txt
- Sample predictions on 20 test images
"""
import os
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# Paths
MODEL_PATH = 'output/yolo_fixed/weights/best.pt'
TEST_IMAGES_DIR = 'dataset_yolo_fixed/images/test'
TEST_LABELS_DIR = 'dataset_yolo_fixed/labels/test'
DATA_YAML = 'dataset_yolo_fixed/data.yaml'
OUTPUT_DIR = 'output/predictions_fixed'
REPORTS_DIR = 'output/reports_fixed'
REPORT_TXT = 'retraining_report.txt'

DAMAGE_NAMES = ['dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage']

# Original metrics (from diagnostic report)
ORIGINAL_METRICS = {
    'mAP50': 0.083,
    'mAP50-95': 0.035,
    'Precision': 0.215,
    'Recall': 0.144,
    'F1': 0.173,
}

def get_test_images():
    """Get list of test images."""
    img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    images = [f for f in os.listdir(TEST_IMAGES_DIR) 
              if any(f.lower().endswith(ext) for ext in img_extensions)]
    return sorted(images)[:20]  # 20 samples

def predict_and_save(model, image_path, output_path, conf_threshold=0.25):
    """Run prediction and save annotated image."""
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=0.5,
        save=True,
        project=str(Path(output_path).parent),
        name=Path(output_path).stem,
        exist_ok=True,
        imgsz=640,
    )
    
    # Get detailed results
    result = results[0]
    detections = []
    if result.boxes is not None:
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class_id': cls_id,
                'class_name': DAMAGE_NAMES[cls_id] if cls_id < len(DAMAGE_NAMES) else 'unknown',
                'confidence': conf,
                'bbox': xyxy,
            })
    
    return detections

def evaluate_model():
    """Run model evaluation on test set and generate metrics."""
    print("=" * 60)
    print("EVALUATING RETRAINED MODEL")
    print("=" * 60)
    
    # Load data config
    with open(DATA_YAML) as f:
        data_cfg = yaml.safe_load(f)
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Run validation
    print("\nRunning validation...")
    val_results = model.val(
        data=DATA_YAML,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.5,
        save_json=True,
        plots=True,
    )
    
    # Extract metrics
    metrics = {
        'mAP50': float(val_results.box.map50),
        'mAP50-95': float(val_results.box.map),
        'Precision': float(val_results.box.mp),
        'Recall': float(val_results.box.mr),
    }
    
    # Calculate F1
    p = metrics['Precision']
    r = metrics['Recall']
    metrics['F1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    # Per-class metrics
    per_class = {}
    if hasattr(val_results.box, 'ap_class_index'):
        for i, cls_id in enumerate(val_results.box.ap_class_index):
            name = data_cfg['names'][cls_id] if cls_id < len(data_cfg['names']) else f'class_{cls_id}'
            per_class[name] = {
                'precision': float(val_results.box.p[i]),
                'recall': float(val_results.box.r[i]),
                'mAP50': float(val_results.box.ap50[i]),
            }
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"mAP50:     {metrics['mAP50']:.4f}")
    print(f"mAP50-95:  {metrics['mAP50-95']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1:        {metrics['F1']:.4f}")
    
    print(f"\nPer-class metrics:")
    for name, cls_metrics in per_class.items():
        print(f"  {name}:")
        for k, v in cls_metrics.items():
            print(f"    {k}: {v:.4f}")
    
    return metrics, per_class, val_results

def generate_report(metrics, per_class):
    """Generate retraining_report.txt"""
    print("\nGenerating retraining report...")
    
    with open(REPORT_TXT, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("YOLOv8 RETRAINING REPORT - Fixed Dataset\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. DATASET SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write("  Dataset: dataset_yolo_fixed/\n")
        f.write("  Classes: 6 damage types only\n")
        f.write("    - dent\n")
        f.write("    - scratch\n")
        f.write("    - crack\n")
        f.write("    - broken_part\n")
        f.write("    - paint_damage\n")
        f.write("    - other_damage\n")
        f.write("  Removed: locations (9 classes) and severity (3 classes)\n")
        f.write("  Fix: Deduplicated triplicate annotations\n\n")
        
        f.write("2. METRICS COMPARISON\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Metric':<20} {'Original (18 cls)':<20} {'Fixed (6 cls)':<20} {'Change':<10}\n")
        f.write("-" * 70 + "\n")
        
        for metric in ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'F1']:
            orig = ORIGINAL_METRICS.get(metric, 0.0)
            new_val = metrics.get(metric, 0.0)
            change = new_val - orig
            change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
            f.write(f"{metric:<20} {orig:<20.4f} {new_val:<20.4f} {change_str:<10}\n")
        
        f.write("\n")
        f.write("3. PER-CLASS PERFORMANCE\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'mAP50':<12}\n")
        f.write("-" * 56 + "\n")
        
        for name, cls_metrics in per_class.items():
            p = cls_metrics.get('precision', 0)
            r = cls_metrics.get('recall', 0)
            m = cls_metrics.get('mAP50', 0)
            f.write(f"{name:<20} {p:<12.4f} {r:<12.4f} {m:<12.4f}\n")
        
        f.write("\n")
        f.write("4. INTERPRETATION\n")
        f.write("-" * 50 + "\n")
        
        if metrics['mAP50'] > 0.3:
            f.write("  ✓ SIGNIFICANT IMPROVEMENT: The fix resolved the core issue.\n")
            f.write("  The model can now learn meaningful damage type distinctions.\n")
        elif metrics['mAP50'] > 0.15:
            f.write("  ✓ MODERATE IMPROVEMENT: The fix helped, but further tuning needed.\n")
        else:
            f.write("  ⚠ LIMITED IMPROVEMENT: Additional issues may remain.\n")
        
        f.write(f"\n  Original mAP50: {ORIGINAL_METRICS['mAP50']:.4f}\n")
        f.write(f"  New mAP50:      {metrics['mAP50']:.4f}\n")
        f.write(f"  Improvement:    {metrics['mAP50'] - ORIGINAL_METRICS['mAP50']:+.4f}\n")
        
        f.write("\n")
        f.write("5. RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n")
        
        # Check for low-performing classes
        low_classes = [n for n, m in per_class.items() if m.get('mAP50', 0) < 0.1]
        if low_classes:
            f.write(f"  ⚠ Low-performing classes: {', '.join(low_classes)}\n")
            f.write("  Consider collecting more training data for these classes.\n")
        
        if metrics['mAP50'] < 0.3:
            f.write("  ⚠ Overall performance still below ideal.\n")
            f.write("  Suggestions:\n")
            f.write("    - Increase epochs (200+)\n")
            f.write("    - Try larger model (yolov8m or yolov8l)\n")
            f.write("    - Add synthetic data for underrepresented classes\n")
            f.write("    - Use GPU for faster training with larger batches\n")
        else:
            f.write("  ✓ Model is ready for initial deployment.\n")
            f.write("  - Fine-tune confidence threshold for production\n")
            f.write("  - Consider adding location/severity as separate models\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Report saved to {REPORT_TXT}")

def generate_sample_predictions(model, test_images):
    """Generate predictions on sample test images."""
    print("\nGenerating sample predictions...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_detections = []
    
    for i, fname in enumerate(test_images):
        img_path = os.path.join(TEST_IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            continue
        
        # Run prediction with save
        results = model.predict(
            source=img_path,
            conf=0.25,
            iou=0.5,
            save=True,
            project=str(Path(OUTPUT_DIR).parent),
            name='predictions_fixed',
            exist_ok=True,
            imgsz=640,
        )
        
        # Get detection details
        result = results[0]
        img_detections = []
        if result.boxes is not None:
            for j in range(len(result.boxes)):
                box = result.boxes[j]
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                
                det = {
                    'image': fname,
                    'class_id': cls_id,
                    'class_name': DAMAGE_NAMES[cls_id] if cls_id < len(DAMAGE_NAMES) else 'unknown',
                    'confidence': conf,
                    'bbox': [round(v, 2) for v in xyxy],
                }
                img_detections.append(det)
                all_detections.append(det)
        
        print(f"  [{i+1}/20] {fname}: {len(img_detections)} detections")
        
        # Rename the saved prediction
        saved_name = fname.rsplit('.', 1)[0] + '.jpg'
        src = os.path.join(OUTPUT_DIR, saved_name)
        # The predict might save as predict.jpg or with original name
        # Let's find the saved file
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith(fname.rsplit('.', 1)[0]) and f.endswith('.jpg'):
                src = os.path.join(OUTPUT_DIR, f)
                break
        
        # Copy to proper name
        dst = os.path.join(OUTPUT_DIR, f'{i+1:04d}_{fname}')
        if os.path.exists(src):
            import shutil
            shutil.copy(src, dst)
    
    # Summary
    print(f"\nPrediction summary:")
    print(f"  Total detections: {len(all_detections)}")
    
    if all_detections:
        df = pd.DataFrame(all_detections)
        print(f"  Average confidence: {df['confidence'].mean():.4f}")
        print(f"  Classes detected: {df['class_name'].value_counts().to_dict()}")
        
        # Confidence distribution
        conf_ranges = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        print(f"  Confidence distribution:")
        for lo, hi in conf_ranges:
            count = len(df[(df['confidence'] >= lo) & (df['confidence'] < hi)])
            if count > 0:
                print(f"    {lo:.2f}-{hi:.2f}: {count}")
    
    # Save detection details as CSV
    if all_detections:
        df = pd.DataFrame(all_detections)
        df.to_csv(os.path.join(OUTPUT_DIR, 'predictions_report.csv'), index=False)
        print(f"\nDetailed predictions saved to {OUTPUT_DIR}/predictions_report.csv")
    
    return all_detections

def check_empty_detections(all_detections, test_images):
    """
    Check how many test images had empty/no detections
    This validates the "significantly fewer empty detections" goal
    """
    detected_images = set(d['image'] for d in all_detections)
    all_images_set = set(test_images)
    empty_images = all_images_set - detected_images
    
    print(f"\nEmpty detection analysis:")
    print(f"  Total test images: {len(test_images)}")
    print(f"  Images with detections: {len(detected_images)}")
    print(f"  Images without detections: {len(empty_images)}")
    if empty_images:
        print(f"  Empty images: {sorted(empty_images)}")
    
    empty_rate = len(empty_images) / len(test_images) * 100 if test_images else 0
    print(f"  Empty detection rate: {empty_rate:.1f}%")
    
    return empty_rate

def main():
    print("=" * 60)
    print("FIXED MODEL EVALUATION PIPELINE")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Get test images
    test_images = get_test_images()
    print(f"\nFound {len(test_images)} test images to evaluate")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Training may still be running or failed.")
        return
    
    model = YOLO(MODEL_PATH)
    
    # Run validation
    metrics, per_class, val_results = evaluate_model()
    
    # Generate retraining report
    generate_report(metrics, per_class)
    
    # Generate sample predictions
    all_detections = generate_sample_predictions(model, test_images)
    
    # Check empty detections
    empty_rate = check_empty_detections(all_detections, test_images)
    
    # Copy validation plots from training output
    import shutil
    train_dir = 'output/yolo_fixed'
    report_dir = REPORTS_DIR
    os.makedirs(report_dir, exist_ok=True)
    
    for fname in ['confusion_matrix.png', 'PR_curve.png', 'results.csv', 'results.png',
                  'P_curve.png', 'R_curve.png', 'F1_curve.png']:
        src = os.path.join(train_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(report_dir, fname))
            print(f"  Copied {fname} to {report_dir}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  mAP50:       {metrics['mAP50']:.4f} (was {ORIGINAL_METRICS['mAP50']:.4f})")
    print(f"  mAP50-95:    {metrics['mAP50-95']:.4f} (was {ORIGINAL_METRICS['mAP50-95']:.4f})")
    print(f"  Precision:   {metrics['Precision']:.4f} (was {ORIGINAL_METRICS['Precision']:.4f})")
    print(f"  Recall:      {metrics['Recall']:.4f} (was {ORIGINAL_METRICS['Recall']:.4f})")
    print(f"  F1:          {metrics['F1']:.4f} (was {ORIGINAL_METRICS['F1']:.4f})")
    print(f"  Empty rate:  {empty_rate:.1f}%")
    print(f"\n  Report:      {REPORT_TXT}")
    print(f"  Predictions: {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()