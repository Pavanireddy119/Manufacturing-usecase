"""
YOLOv8 Performance Diagnostic Tool
Diagnoses why YOLOv8 returns mostly empty detections
"""
import os
import csv
import json
import random
import numpy as np
from collections import Counter, defaultdict
from glob import glob
import shutil
import sys

# Force unbuffered output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

print("=" * 80)
print("YOLOv8 PERFORMANCE DIAGNOSTIC TOOL")
print("=" * 80)

# 1. ANALYZE TRAINING RESULTS
print("\n" + "=" * 80)
print("1. TRAINING RESULTS ANALYSIS")
print("=" * 80)

results_path = 'output/models/results.csv'
if os.path.exists(results_path):
    with open(results_path) as f:
        reader = csv.DictReader(f)
        final_row = None
        all_rows = []
        # CSV has spaces in header - strip them
        fieldnames = [f.strip() for f in open(results_path).readline().strip().split(',')]
        reader = csv.DictReader(open(results_path), fieldnames=fieldnames)
        next(reader)  # Skip first data row if it was the header
        
        for row in reader:
            all_rows.append(row)
            final_row = row
        
        if final_row:
            print(f"\nFinal Epoch Results (Epoch {final_row['epoch'].strip()}):")
            print(f"  mAP50:     {float(final_row['metrics/mAP50(B)'].strip()):.4f}")
            print(f"  mAP50-95:  {float(final_row['metrics/mAP50-95(B)'].strip()):.4f}")
            print(f"  Precision: {float(final_row['metrics/precision(B)'].strip()):.4f}")
            print(f"  Recall:    {float(final_row['metrics/recall(B)'].strip()):.4f}")
            
            # Check best epoch
            best_epoch = max(all_rows, key=lambda r: float(r['metrics/mAP50(B)'].strip()))
            print(f"\nBest Epoch (Epoch {best_epoch['epoch'].strip()}):")
            print(f"  mAP50:     {float(best_epoch['metrics/mAP50(B)'].strip()):.4f}")
            print(f"  mAP50-95:  {float(best_epoch['metrics/mAP50-95(B)'].strip()):.4f}")
            print(f"  Precision: {float(best_epoch['metrics/precision(B)'].strip()):.4f}")
            print(f"  Recall:    {float(best_epoch['metrics/recall(B)'].strip()):.4f}")
            
            # Calculate F1 Score
            p = float(best_epoch['metrics/precision(B)'].strip())
            r = float(best_epoch['metrics/recall(B)'].strip())
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            print(f"  F1 Score:  {f1:.4f}")
            
            # Check if model is just predicting majority class
            print(f"\n  - Model is predicting {len(all_rows)} unique classes?")
            print(f"  - mAP50 trend: first={float(all_rows[0]['metrics/mAP50(B)'].strip()):.4f}, best={float(best_epoch['metrics/mAP50(B)'].strip()):.4f}")
            print(f"  - mAP50-95 trend: first={float(all_rows[0]['metrics/mAP50-95(B)'].strip()):.4f}, best={float(best_epoch['metrics/mAP50-95(B)'].strip()):.4f}")
            
            # Determine if underfitting
            print(f"\nVerdict:")
            if float(best_epoch['metrics/mAP50(B)']) < 0.5:
                print(f"  ❌ Model is UNDERFITTING (mAP50 < 0.5)")
            else:
                print(f"  ✓ Model is learning adequately")
else:
    print("No training results found!")

# 2. ANALYZE DATASET QUALITY
print("\n" + "=" * 80)
print("2. DATASET QUALITY ANALYSIS")
print("=" * 80)

def analyze_labels(label_dir):
    if not os.path.exists(label_dir):
        return {}, 0, 0, []
    
    class_counts = Counter()
    empty_files = 0
    total_annotations = 0
    total_files = 0
    box_areas = []
    
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        total_files += 1
        fpath = os.path.join(label_dir, fname)
        with open(fpath) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        if not lines:
            empty_files += 1
            continue
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_counts[class_id] += 1
                total_annotations += 1
                # Calculate box area
                _, _, w, h = map(float, parts[1:5])
                box_areas.append(w * h)
    
    return class_counts, total_annotations, empty_files, box_areas

# Check all label directories
label_dirs = [
    ('Training', 'dataset_final/labels/train'),
    ('Validation', 'dataset_final/labels/val'),
    ('Test', 'dataset_final/labels/test')
]

all_class_counts = Counter()
total_annotations_all = 0
total_empty = 0
all_box_areas = []

for split_name, label_dir in label_dirs:
    counts, n_ann, n_empty, areas = analyze_labels(label_dir)
    print(f"\n{split_name} Labels ({label_dir}):")
    print(f"  Files with labels: {len(os.listdir(label_dir)) if os.path.exists(label_dir) else 0}")
    print(f"  Total annotations: {n_ann}")
    print(f"  Empty label files: {n_empty}")
    print(f"  Unique classes: {len(counts)}")
    all_class_counts.update(counts)
    total_annotations_all += n_ann
    total_empty += n_empty
    all_box_areas.extend(areas)

# Class distribution
print(f"\nClass Distribution (Total):")
class_names = ['dent','scratch','crack','broken_part','paint_damage','other_damage',
               'front_bumper','rear_bumper','hood','windshield','left_door','right_door',
               'roof','side_panel','other_location','low','medium','high']
for class_id in sorted(all_class_counts.keys()):
    name = class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
    count = all_class_counts[class_id]
    bar = '█' * min(count, 50)
    print(f"  {class_id:2d} {name:20s}: {count:4d} {bar}")

# Box size analysis
if all_box_areas:
    print(f"\nBox Size Analysis:")
    print(f"  Min area: {min(all_box_areas):.6f}")
    print(f"  Max area: {max(all_box_areas):.6f}")
    print(f"  Mean area: {np.mean(all_box_areas):.6f}")
    print(f"  Median area: {np.median(all_box_areas):.6f}")
    small_boxes = sum(1 for a in all_box_areas if a < 0.01)
    print(f"  Very small boxes (< 1% image): {small_boxes}/{len(all_box_areas)} ({100*small_boxes/len(all_box_areas):.1f}%)")

# 3. CLASS STRUCTURE ANALYSIS
print("\n" + "=" * 80)
print("3. CLASS STRUCTURE ANALYSIS")
print("=" * 80)

print("""
Current Mixed Classes:
  Classes 0-5:  Damage Types (dent, scratch, crack, broken_part, paint_damage, other_damage)
  Classes 6-14: Locations (front_bumper, rear_bumper, hood, windshield, left_door, right_door, roof, side_panel, other_location)
  Classes 15-17: Severity (low, medium, high)

CRITICAL ISSUE: YOLO is being trained on MIXED class types!
- A 'dent' (class 0) is trained in the same space as 'front_bumper' (class 6)
- These are fundamentally different semantic concepts
- The model cannot learn to distinguish dent from front_bumper because
  they're treated as distinct categories rather than attributes
""")

# Count per class group
damage_types = sum(all_class_counts.get(i, 0) for i in range(6))
locations = sum(all_class_counts.get(i, 0) for i in range(6, 15))
severities = sum(all_class_counts.get(i, 0) for i in range(15, 18))
print(f"\nClass Group Distribution:")
print(f"  Damage Types: {damage_types} ({100*damage_types/total_annotations_all:.1f}%)" if total_annotations_all > 0 else "  Damage Types: 0")
print(f"  Locations:     {locations} ({100*locations/total_annotations_all:.1f}%)" if total_annotations_all > 0 else "  Locations: 0")
print(f"  Severities:    {severities} ({100*severities/total_annotations_all:.1f}%)" if total_annotations_all > 0 else "  Severities: 0")

# 4. CHECK EXISTING INFERENCE CODE
print("\n" + "=" * 80)
print("4. PREDICTION THRESHOLD TESTING SCRIPT")
print("=" * 80)

with open('evaluate_yolo.py') as f:
    eval_code = f.read()
    print("\nCurrent evaluate_yolo.py - checking inference parameters...")
    if 'conf=' in eval_code:
        print("  ✓ Found confidence threshold parameter")
    else:
        print("  ✗ No confidence threshold set (using default 0.25)")
    if 'predict' in eval_code:
        print("  ✓ Found predict() call")

# 5. DATASET STATISTICS
print("\n" + "=" * 80)
print("5. DATASET SIZE ANALYSIS")
print("=" * 80)

for split_name, img_dir in [('Train', 'dataset_final/images/train'), 
                              ('Val', 'dataset_final/images/val'),
                              ('Test', 'dataset_final/images/test')]:
    if os.path.exists(img_dir):
        n_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"  {split_name}: {n_imgs} images")
    
for split_name, label_dir in [('Train', 'dataset_final/labels/train'),
                               ('Val', 'dataset_final/labels/val'),
                               ('Test', 'dataset_final/labels/test')]:
    if os.path.exists(label_dir):
        n_labels = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
        print(f"  {split_name} labels: {n_labels}")

# Check for images without labels
print("\nChecking for images without corresponding labels...")
for split_name, img_dir, label_dir in [
    ('Train', 'dataset_final/images/train', 'dataset_final/labels/train'),
    ('Val', 'dataset_final/images/val', 'dataset_final/labels/val')
]:
    if os.path.exists(img_dir) and os.path.exists(label_dir):
        img_files = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png')))
        label_files = set(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt'))
        missing_labels = img_files - label_files
        extra_labels = label_files - img_files
        if missing_labels:
            print(f"  {split_name}: {len(missing_labels)} images missing labels")
        if extra_labels:
            print(f"  {split_name}: {len(extra_labels)} labels without images")

# 6. ROOT CAUSE SUMMARY
print("\n" + "=" * 80)
print("6. ROOT CAUSE ANALYSIS")
print("=" * 80)

issues_found = []

# A. Check annotations
if total_annotations_all == 0:
    issues_found.append("A. No annotations found in dataset!")
elif total_empty > 0:
    issues_found.append(f"A. {total_empty} empty label files found")

# B. Check class imbalance
if all_class_counts:
    max_count = max(all_class_counts.values())
    min_count = min(all_class_counts.values())
    if max_count > 10 * min_count:
        issues_found.append(f"B. Severe class imbalance (max:{max_count} vs min:{min_count})")

# C. Mixed class structure
issues_found.append("C. CRITICAL: Mixed class structure with damage types + locations + severity")
issues_found.append("   This alone can cause the model to fail to learn meaningful features")

# D. Check training epochs
if os.path.exists(results_path):
    with open(results_path) as f:
        n_epochs = sum(1 for _ in csv.DictReader(f))
    if n_epochs < 50:
        issues_found.append(f"D. Insufficient training ({n_epochs} epochs)")

# E. Dataset size
total_images = 0
for img_dir in ['dataset_final/images/train', 'dataset_final/images/val']:
    if os.path.exists(img_dir):
        total_images += len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
if total_images < 200:
    issues_found.append(f"F. Dataset too small ({total_images} images total)")

print("\nIssues Found:")
for issue in issues_found:
    print(f"  • {issue}")

if not issues_found:
    print("  ✓ No obvious issues found")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)