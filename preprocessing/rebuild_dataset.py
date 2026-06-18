"""
Complete rebuild of YOLOv8 dataset: dataset_yolo_fixed/
- Remove triplicate annotations (damage type + location + severity)
- Keep ONLY damage type classes (0-5)
- Deduplicate bounding boxes
- Generate proper data.yaml
- Full validation
"""
import os
import shutil
from collections import defaultdict, Counter

SOURCE_DIR = 'datasets/dataset_final'
OUTPUT_DIR = 'datasets/dataset_yolo_fixed'

DAMAGE_NAMES = ['dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage']
DAMAGE_CLASSES = set(range(6))  # Classes 0-5 are damage types

def rebuild_dataset():
    """Create datasets/dataset_yolo_fixed/ with deduplicated damage-type-only annotations."""
    
    print("=" * 60)
    print("REBUILDING DATASET: datasets/dataset_yolo_fixed/")
    print("=" * 60)
    
    # Clean output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    # Create directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{OUTPUT_DIR}/images/{split}', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/labels/{split}', exist_ok=True)
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        img_src = f'{SOURCE_DIR}/images/{split}'
        lbl_src = f'{SOURCE_DIR}/labels/{split}'
        
        if not os.path.exists(img_src):
            print(f"  Warning: {img_src} not found, skipping")
            continue
        
        print(f"\nProcessing {split} split...")
        
        # Get image files
        img_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        image_files = [f for f in os.listdir(img_src) 
                       if any(f.lower().endswith(ext) for ext in img_extensions)]
        
        stats[split] = {'images': 0, 'annotations': 0, 'empty': 0}
        class_counts = Counter()
        total_duplicates_removed = 0
        
        for fname in sorted(image_files):
            base = os.path.splitext(fname)[0]
            src_img = os.path.join(img_src, fname)
            src_lbl = os.path.join(lbl_src, f'{base}.txt')
            
            if not os.path.exists(src_lbl):
                continue
            
            # Copy image
            dst_img = os.path.join(f'{OUTPUT_DIR}/images/{split}', fname)
            shutil.copy2(src_img, dst_img)
            stats[split]['images'] += 1
            
            # Read labels and extract only damage types (classes 0-5)
            damage_annotations = {}
            with open(src_lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    
                    # Keep only damage type classes
                    if class_id in DAMAGE_CLASSES:
                        # Round coordinates to 4 decimal places for dedup
                        x, y, w, h = [round(float(v), 4) for v in parts[1:]]
                        box_key = (x, y, w, h)
                        
                        # If this box already seen with a different damage type, 
                        # keep the first one (they're all the same box duplicated)
                        if box_key not in damage_annotations:
                            damage_annotations[box_key] = f"{class_id} {x} {y} {w} {h}"
                        else:
                            total_duplicates_removed += 1
            
            # Write deduplicated labels
            dst_lbl = os.path.join(f'{OUTPUT_DIR}/labels/{split}', f'{base}.txt')
            if damage_annotations:
                with open(dst_lbl, 'w') as f:
                    for box_key in sorted(damage_annotations.keys()):
                        line = damage_annotations[box_key]
                        f.write(line + '\n')
                        cls = int(line.split()[0])
                        class_counts[cls] += 1
                        stats[split]['annotations'] += 1
            else:
                # Empty label file (image with no damage annotations)
                open(dst_lbl, 'w').close()
                stats[split]['empty'] += 1
        
        print(f"  Images: {stats[split]['images']}")
        print(f"  Damage annotations: {stats[split]['annotations']}")
        print(f"  Empty (no damage): {stats[split]['empty']}")
        print(f"  Duplicates removed: {total_duplicates_removed}")
        
        # Print class distribution
        print(f"  Class distribution:")
        for cls_id in range(6):
            if cls_id in class_counts:
                print(f"    {DAMAGE_NAMES[cls_id]} (ID {cls_id}): {class_counts[cls_id]}")
            else:
                print(f"    {DAMAGE_NAMES[cls_id]} (ID {cls_id}): 0")
    
    # Create data.yaml
    import yaml
    abs_path = os.path.abspath(OUTPUT_DIR)
    data_yaml = {
        'path': abs_path,
        'train': f'{abs_path}/images/train',
        'val': f'{abs_path}/images/val',
        'test': f'{abs_path}/images/test',
        'nc': 6,
        'names': DAMAGE_NAMES,
    }
    
    with open(f'{OUTPUT_DIR}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\ndata.yaml created at {OUTPUT_DIR}/data.yaml")
    print(f"Dataset rebuild complete!")
    
    return stats

def validate_dataset():
    """Validate no duplicate boxes, no invalid classes, no empty issues."""
    print("\n" + "=" * 60)
    print("VALIDATING FIXED DATASET")
    print("=" * 60)
    
    issues = []
    
    for split in ['train', 'val', 'test']:
        lbl_dir = f'{OUTPUT_DIR}/labels/{split}'
        if not os.path.exists(lbl_dir):
            continue
        
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            
            filepath = os.path.join(lbl_dir, fname)
            with open(filepath) as f:
                lines = [l.strip() for l in f if l.strip()]
            
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    issues.append(f"{split}/{fname}:{i+1} - Wrong format: {len(parts)} fields")
                    continue
                
                class_id = int(parts[0])
                if class_id < 0 or class_id > 5:
                    issues.append(f"{split}/{fname}:{i+1} - Invalid class {class_id}")
                
                coords = [float(v) for v in parts[1:]]
                if not all(0 <= v <= 1 for v in coords):
                    issues.append(f"{split}/{fname}:{i+1} - Coords out of range: {coords}")
            
            # Check for duplicate boxes within the same file
            seen = set()
            for i, line in enumerate(lines):
                parts = line.split()
                class_id = int(parts[0])
                coords = tuple(round(float(v), 4) for v in parts[1:])
                key = (class_id, coords)
                if key in seen:
                    issues.append(f"{split}/{fname}:{i+1} - Duplicate box (class {class_id}, {coords})")
                seen.add(key)
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\n✓ No issues found! Dataset is clean.")
    
    # Summary stats
    total_annotations = 0
    total_images = 0
    class_counts_total = Counter()
    
    for split in ['train', 'val', 'test']:
        lbl_dir = f'{OUTPUT_DIR}/labels/{split}'
        img_dir = f'{OUTPUT_DIR}/images/{split}'
        if not os.path.exists(lbl_dir):
            continue
        
        images = [f for f in os.listdir(img_dir) if any(f.lower().endswith(e) for e in ('.jpg', '.jpeg', '.png'))]
        total_images += len(images)
        
        for fname in os.listdir(lbl_dir):
            if not fname.endswith('.txt'):
                continue
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    if line.strip():
                        cls = int(line.strip().split()[0])
                        class_counts_total[cls] += 1
                        total_annotations += 1
    
    print(f"\nDataset Summary:")
    print(f"  Total images: {total_images}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Class distribution:")
    for cls_id in range(6):
        name = DAMAGE_NAMES[cls_id]
        count = class_counts_total.get(cls_id, 0)
        print(f"    {name} (ID {cls_id}): {count}")
    
    return len(issues) == 0

if __name__ == '__main__':
    # Step 1: Rebuild dataset
    stats = rebuild_dataset()
    
    # Step 2: Validate
    is_valid = validate_dataset()
    
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT:")
    if is_valid:
        print("✓ Dataset is ready for training!")
        print(f"  Path: {os.path.abspath(OUTPUT_DIR)}")
        print(f"  Config: {os.path.abspath(OUTPUT_DIR)}/data.yaml")
    else:
        print("✗ Dataset has issues that need fixing.")
