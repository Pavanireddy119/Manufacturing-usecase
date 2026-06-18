"""
Comprehensive Root Cause Analysis and Model Evaluation
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict
import json

def load_yaml(yaml_path):
    """Load YAML config"""
    import yaml
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model():
    """Evaluate the trained model"""
    
    print("=" * 80)
    print("MODEL EVALUATION AND ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    # Load model
    model_path = "outputs/model_outputs/runs/detect/damage_detection_v1/weights/best.pt"
    model = YOLO(model_path)
    
    # Dataset paths
    dataset_path = Path("datasets/dataset_cleaned")
    test_images_dir = dataset_path / 'images' / 'test'
    test_labels_dir = dataset_path / 'labels' / 'test'
    
    # Get classes
    with open(dataset_path / 'classes.txt', 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    print(f"\nClasses: {classes}")
    print(f"Model: {model_path}")
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    
    if test_images_dir.exists():
        test_images = sorted([f for f in os.listdir(test_images_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nTest images: {len(test_images)}")
        
        all_predictions = []
        all_ground_truth = []
        empty_predictions = 0
        
        for img_file in test_images:
            img_path = test_images_dir / img_file
            label_file = test_labels_dir / (img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            print(f"\n[{img_file}]")
            
            # Load image
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            # Get predictions
            results = model.predict(source=str(img_path), conf=0.1, verbose=False)
            
            # Extract predictions
            detections = results[0].boxes
            if detections is None or len(detections) == 0:
                print(f"  PREDICTIONS: EMPTY (no detections)")
                empty_predictions += 1
            else:
                print(f"  PREDICTIONS: {len(detections)} detections")
                for det in detections:
                    class_id = int(det.cls[0].item())
                    conf = det.conf[0].item()
                    bbox = det.xyxy[0].cpu().numpy()
                    print(f"    - {classes[class_id]}: conf={conf:.3f}, bbox={bbox}")
                    all_predictions.append({
                        'class': classes[class_id],
                        'conf': conf,
                        'bbox': bbox.tolist()
                    })
            
            # Get ground truth
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                
                if lines:
                    print(f"  GROUND TRUTH: {len(lines)} annotations")
                    for line in lines:
                        parts = line.split()
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        px_left = (x_center - width/2) * w
                        px_top = (y_center - height/2) * h
                        px_right = (x_center + width/2) * w
                        px_bottom = (y_center + height/2) * h
                        
                        print(f"    - {classes[class_id]}: bbox=[{px_left:.0f}, {px_top:.0f}, {px_right:.0f}, {px_bottom:.0f}]")
                        all_ground_truth.append({
                            'class': classes[class_id],
                            'bbox': [px_left, px_top, px_right, px_bottom]
                        })
                else:
                    print(f"  GROUND TRUTH: EMPTY")
        
        print(f"\n\nTEST SET SUMMARY:")
        print(f"  Total test images: {len(test_images)}")
        print(f"  Empty predictions: {empty_predictions}/{len(test_images)} ({100*empty_predictions/len(test_images):.1f}%)")
        print(f"  Total predictions: {len(all_predictions)}")
        print(f"  Total ground truth: {len(all_ground_truth)}")
    
    # Root cause analysis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS: WHY THE MODEL FAILED")
    print("=" * 80)
    
    report = []
    report.append("\nREASON FOR FAILURE: Multiple Critical Issues\n")
    
    # Issue 1: Dataset size
    report.append("1. DATASET SIZE - CRITICAL LIMITATION")
    report.append("-" * 80)
    report.append(f"   Training images: 36")
    report.append(f"   Validation images: 7")
    report.append(f"   Test images: 2")
    report.append(f"   Total: 45 images")
    report.append(f"")
    report.append(f"   PROBLEM: YOLOv8n has 3.01M parameters. Training on 36 images")
    report.append(f"   causes severe OVERFITTING and UNDERFITTING.")
    report.append(f"")
    report.append(f"   BENCHMARK: YOLOv8n typically requires 1000+ images minimum")
    report.append(f"   Recommended: 5000+ images for production model")
    report.append(f"")
    report.append(f"   IMPACT: Model cannot learn generalizable features")
    report.append(f"")
    
    # Issue 2: Label quality
    report.append("2. LABEL QUALITY - HIGH IMPACT")
    report.append("-" * 80)
    report.append(f"   Original dataset had:")
    report.append(f"     - 22 images with EMPTY labels (no annotations)")
    report.append(f"     - 18 classes instead of 5 target classes")
    report.append(f"     - 0 samples for 'scratch' class")
    report.append(f"     - Mix of damage types, locations, and severity as single classes")
    report.append(f"")
    report.append(f"   Impact on model:")
    report.append(f"     - Mixed labels confuse the model")
    report.append(f"     - Auto-labeling errors propagate during training")
    report.append(f"     - Missing classes (scratch) prevent complete learning")
    report.append(f"")
    
    # Issue 3: Class imbalance
    report.append("3. CLASS IMBALANCE - HIGH IMPACT")
    report.append("-" * 80)
    report.append(f"   Current distribution:")
    report.append(f"     - Crack: 49 annotations (44%)")
    report.append(f"     - Dent: 46 annotations (41%)")
    report.append(f"     - Paint damage: 9 annotations (8%)")
    report.append(f"     - Broken part: 7 annotations (6%)")
    report.append(f"     - Scratch: 0 annotations (0%) MISSING CLASS")
    report.append(f"")
    report.append(f"   Imbalance ratio: 49:7 = 7:1 (SEVERE)")
    report.append(f"")
    report.append(f"   Impact:")
    report.append(f"     - Model biased towards crack and dent")
    report.append(f"     - Poor detection of minority classes")
    report.append(f"     - Cannot predict scratch at all")
    report.append(f"")
    
    # Issue 4: Model architecture
    report.append("4. MODEL SIZE MISMATCH - MODERATE IMPACT")
    report.append("-" * 80)
    report.append(f"   Model: YOLOv8n")
    report.append(f"   Parameters: 3.01 Million")
    report.append(f"   Training samples: 36")
    report.append(f"")
    report.append(f"   Parameters-to-samples ratio: 3.01M / 36 = 83,611:1 !!!")
    report.append(f"")
    report.append(f"   This extreme ratio causes:")
    report.append(f"     - Severe overfitting")
    report.append(f"     - Model memorizes training data")
    report.append(f"     - Poor generalization to test data")
    report.append(f"")
    
    # Issue 5: Training dynamics
    report.append("5. TRAINING STOPPED EARLY - SYMPTOM")
    report.append("-" * 80)
    report.append(f"   Configured epochs: 30")
    report.append(f"   Actual epochs: 6")
    report.append(f"   Reason: Early stopping (patience=5)")
    report.append(f"")
    report.append(f"   Why it stopped:")
    report.append(f"     - Validation mAP50 not improving after 5 epochs")
    report.append(f"     - Model not learning meaningful patterns")
    report.append(f"     - Random initialization + overfitting = no improvement")
    report.append(f"")
    
    # Diagnosis
    report.append("\n" + "=" * 80)
    report.append("DIAGNOSIS")
    report.append("=" * 80)
    
    report.append(f"\nModel Performance Metrics (FAILED):")
    report.append(f"  - mAP50: 0.002 (Target: 0.50+)")
    report.append(f"  - Precision: 0.001 (Target: 0.80+)")
    report.append(f"  - Recall: 0.053 (Target: 0.80+)")
    report.append(f"  - mAP50-95: 0.001 (Target: 0.35+)")
    report.append(f"")
    report.append(f"Primary Cause: INSUFFICIENT DATA (45 images << 1000 minimum)")
    report.append(f"")
    report.append(f"Contributing factors:")
    report.append(f"  1. Auto-labeled data with quality issues")
    report.append(f"  2. Severe class imbalance (missing 'scratch' class)")
    report.append(f"  3. Mixed classification scheme (18 classes vs 5 target)")
    report.append(f"  4. Early stopping due to no improvement")
    report.append(f"")
    
    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("RECOMMENDATIONS TO FIX THE MODEL")
    report.append("=" * 80)
    
    report.append(f"\n1. DATA COLLECTION (PRIMARY PRIORITY)")
    report.append(f"   - Collect minimum 1000-2000 images per damage type")
    report.append(f"   - Ensure balanced distribution across all 5 classes")
    report.append(f"   - MANDATORY: Add 'scratch' samples (currently 0)")
    report.append(f"   - Manual annotation or professional labeling service")
    report.append(f"")
    
    report.append(f"2. LABEL QUALITY ASSURANCE")
    report.append(f"   - Manual review of all auto-generated labels")
    report.append(f"   - Remove/fix incorrect annotations")
    report.append(f"   - Implement inter-annotator agreement checks")
    report.append(f"   - Remove mislabeled images")
    report.append(f"")
    
    report.append(f"3. CLASS BALANCING")
    report.append(f"   - Augment minority classes (paint_damage, broken_part)")
    report.append(f"   - Use oversampling for underrepresented damage types")
    report.append(f"   - Implement weighted loss during training")
    report.append(f"   - Ensure at least 200-300 samples per class minimum")
    report.append(f"")
    
    report.append(f"4. TRAINING STRATEGY")
    report.append(f"   - After collecting more data, retrain for 100-200 epochs")
    report.append(f"   - Use transfer learning (ImageNet pretrained weights)")
    report.append(f"   - Implement learning rate scheduling")
    report.append(f"   - Use data augmentation (mosaic, mixup, rotation, etc.)")
    report.append(f"")
    
    report.append(f"5. MODEL SELECTION")
    report.append(f"   - Consider smaller model (yolov8n or yolov8s)")
    report.append(f"   - With limited data, prefer simpler models")
    report.append(f"   - Once data grows to 2000+, use YOLOv8m")
    report.append(f"")
    
    # Print and save report
    full_report = "\n".join(report)
    print(full_report)
    
    # Save to file
    report_path = Path("root_cause_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(full_report)
    
    print(f"\n[OK] Root cause analysis saved: {report_path}")
    
    return report_path

if __name__ == "__main__":
    evaluate_model()
