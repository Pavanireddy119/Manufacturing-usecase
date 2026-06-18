"""
Restructure the dataset for YOLOv8 training.
The critical fix: Each bounding box gets ONE class label.
Remove triplicate overlapping boxes for damage type + location + severity.
"""
import os
import shutil
import random
from collections import defaultdict, Counter
import cv2
import numpy as np

# Paths
BASE_DIR = 'datasets/dataset_final'
OUTPUT_DIR = 'datasets/dataset_yolo_restructured'

# Clean output
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Create directories
for split in ['train', 'val']:
    os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)

# Strategy: Restructure to 3 separate models
# We'll create specialized datasets for each task

# DATA STRUCTURE:
# Each annotation line in current format: class_id x_center y_center width height
# Current classes 0-5: damage types (dent, scratch, crack, broken_part, paint_damage, other_damage)
# Current classes 6-14: locations
# Current classes 15-17: severity

# Strategy 1: Single "damage" detection (binary - damage vs no damage)
# Strategy 2: Damage type classification (6 classes)
# Strategy 3: Location + severity as separate data

# We'll implement Approach A: Train YOLO on damage types only (simplest, fastest path to results)
# Then post-process to add location/severity metadata

def create_damage_type_dataset():
    """Create dataset with only damage type classes (0-5)"""
    
    print("Creating Damage-Type-YOLO dataset...")
    
    damage_names = ['dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage']
    
    for split in ['train', 'val']:
        img_dir = f'{BASE_DIR}/images/{split}'
        lbl_dir = f'{BASE_DIR}/labels/{split}'
        
        if not os.path.exists(img_dir):
            continue
            
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            base = os.path.splitext(fname)[0]
            src_img = os.path.join(img_dir, fname)
            src_lbl = os.path.join(lbl_dir, f'{base}.txt')
            
            if not os.path.exists(src_lbl):
                continue
            
            # Copy image
            dst_img = os.path.join(f'{OUTPUT_DIR}/images/{split}', fname)
            shutil.copy2(src_img, dst_img)
            
            # Process labels - keep only damage types, remove duplicates
            damage_annotations = []
            seen_boxes = set()  # Track unique box positions
            
            with open(src_lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    
                    # Only keep damage type classes (0-5)
                    if 0 <= class_id <= 5:
                        # Round coordinates to avoid near-duplicates
                        box_key = tuple(round(float(x), 4) for x in parts[1:])
                        if box_key not in seen_boxes:
                            seen_boxes.add(box_key)
                            damage_annotations.append(line.strip())
            
            # Write filtered labels
            dst_lbl = os.path.join(f'{OUTPUT_DIR}/labels/{split}', f'{base}.txt')
            if damage_annotations:
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(damage_annotations))
            else:
                # Create empty label file (image with no damage)
                open(dst_lbl, 'w').close()
    
    # Create data.yaml
    yaml_content = f"""names:
- dent
- scratch
- crack
- broken_part
- paint_damage
- other_damage
nc: 6
path: {os.path.abspath(OUTPUT_DIR)}
train: {os.path.abspath(OUTPUT_DIR)}/images/train
val: {os.path.abspath(OUTPUT_DIR)}/images/val
"""
    with open(f'{OUTPUT_DIR}/damage_only.yaml', 'w') as f:
        f.write(yaml_content)

def create_single_damage_dataset():
    """Create dataset with single 'damage' class - just detect if damage exists"""
    
    print("Creating Single-Class-Damage dataset...")
    
    for split in ['train', 'val']:
        img_dir = f'{BASE_DIR}/images/{split}'
        lbl_dir = f'{BASE_DIR}/labels/{split}'
        
        if not os.path.exists(img_dir):
            continue
            
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            base = os.path.splitext(fname)[0]
            src_img = os.path.join(img_dir, fname)
            src_lbl = os.path.join(lbl_dir, f'{base}.txt')
            
            if not os.path.exists(src_lbl):
                continue
            
            # Process labels - merge all damage into single class
            damage_boxes = []
            seen_boxes = set()
            
            with open(src_lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    
                    # Only keep damage types (0-5)
                    if 0 <= class_id <= 5:
                        box_key = tuple(round(float(x), 4) for x in parts[1:])
                        if box_key not in seen_boxes:
                            seen_boxes.add(box_key)
                            damage_boxes.append(parts[1:])  # [x, y, w, h]
            
            # Write labels with class 0 for all damage
            dst_lbl_dir = f'{OUTPUT_DIR}/labels/{split}'
            os.makedirs(dst_lbl_dir, exist_ok=True)
            dst_lbl = os.path.join(dst_lbl_dir, f'{base}.txt')
            
            if damage_boxes:
                lines = []
                for box in damage_boxes:
                    lines.append(f"0 {' '.join(box)}")
                with open(dst_lbl, 'w') as f:
                    f.write('\n'.join(lines))
            else:
                open(dst_lbl, 'w').close()

def analyze_new_dataset():
    """Verify the restructured dataset"""
    print("\n=== NEW DATASET ANALYSIS ===")
    
    for split in ['train', 'val']:
        lbl_dir = f'{OUTPUT_DIR}/labels/{split}'
        if not os.path.exists(lbl_dir):
            continue
            
        c = Counter()
        total_files = 0
        empty_files = 0
        
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            total_files += 1
            with open(os.path.join(lbl_dir, fname)) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                empty_files += 1
            for line in lines:
                cls = int(line.split()[0])
                c[cls] += 1
        
        print(f"\n{split}:")
        print(f"  Files: {total_files}")
        print(f"  Empty: {empty_files}")
        print(f"  Annotations: {sum(c.values())}")
        print(f"  Classes used: {len(c)}")
        names = ['dent','scratch','crack','broken_part','paint_damage','other_damage','damage']
        for cls in sorted(c.keys()):
            name = names[cls] if cls < len(names) else 'unknown'
            print(f"    Class {cls} ({name}): {c[cls]}")

if __name__ == '__main__':
    # Create damage-only dataset (6 classes)
    create_damage_type_dataset()
    
    # Create single damage class dataset (1 class)
    # We'll use the damage-only one as primary
    
    analyze_new_dataset()
    
    print(f"\nDataset created at: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Config file: {os.path.abspath(OUTPUT_DIR)}/damage_only.yaml")
