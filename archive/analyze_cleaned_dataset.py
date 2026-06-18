"""
Analyze cleaned dataset class distribution
"""

import os
from pathlib import Path
from collections import defaultdict
import cv2

def analyze_cleaned_dataset():
    dataset_path = Path(r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\dataset_cleaned")
    classes_file = dataset_path / "classes.txt"
    
    # Load classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    class_stats = defaultdict(lambda: {'annotations': 0, 'images': set(), 'bbox_sizes': []})
    split_stats = {}
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        split_images = 0
        split_annotations = 0
        
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_file in image_files:
            img_path = images_dir / img_file
            label_file = labels_dir / (img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            if not label_file.exists():
                continue
            
            split_images += 1
            
            # Get image size
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            
            # Read annotations
            with open(label_file, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            
            for line in lines:
                parts = line.split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert to pixels
                px_w = width * img_w
                px_h = height * img_h
                
                class_name = classes[class_id]
                class_stats[class_name]['annotations'] += 1
                class_stats[class_name]['images'].add(img_file)
                class_stats[class_name]['bbox_sizes'].append((px_w, px_h))
                split_annotations += 1
        
        split_stats[split] = {'images': split_images, 'annotations': split_annotations}
    
    # Print report
    report_path = Path(r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase") / "class_distribution_cleaned.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLEANED DATASET CLASS DISTRIBUTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        # Split summary
        f.write("SPLIT SUMMARY:\n")
        f.write("-" * 80 + "\n")
        total_images = 0
        total_annotations = 0
        for split in ['train', 'val', 'test']:
            if split in split_stats:
                stats = split_stats[split]
                f.write(f"{split.upper():10s}: {stats['images']:3d} images, {stats['annotations']:4d} annotations\n")
                total_images += stats['images']
                total_annotations += stats['annotations']
        f.write(f"\nTOTAL     : {total_images:3d} images, {total_annotations:4d} annotations\n")
        
        # Class distribution
        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Annotations':>12} {'Images':>8} {'Avg Size':>12} {'Imbalance':>12}\n")
        f.write("-" * 80 + "\n")
        
        max_count = max(stat['annotations'] for stat in class_stats.values()) if class_stats else 1
        
        for class_name in classes:
            stats = class_stats[class_name]
            count = stats['annotations']
            images = len(stats['images'])
            
            if count > 0:
                bbox_sizes = stats['bbox_sizes']
                avg_w = sum(s[0] for s in bbox_sizes) / len(bbox_sizes)
                avg_h = sum(s[1] for s in bbox_sizes) / len(bbox_sizes)
                avg_size = f"{avg_w:.0f}x{avg_h:.0f}"
                
                imbalance = max_count / count if count > 0 else 0
                imbalance_str = f"{imbalance:.1f}x"
            else:
                avg_size = "N/A"
                imbalance_str = "INF"
            
            f.write(f"{class_name:<20} {count:>12} {images:>8} {avg_size:>12} {imbalance_str:>12}\n")
        
        # Imbalance analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("CLASS IMBALANCE ANALYSIS:\n")
        f.write("-" * 80 + "\n")
        
        counts = [stat['annotations'] for stat in class_stats.values() if stat['annotations'] > 0]
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else 0
            f.write(f"Max annotations: {max_count}\n")
            f.write(f"Min annotations: {min_count}\n")
            f.write(f"Imbalance ratio: {imbalance_ratio:.1f}:1\n")
            
            if imbalance_ratio > 5:
                f.write(f"\nWARNING - HIGH IMBALANCE: Consider using class weighting or augmentation\n")
            elif imbalance_ratio > 2:
                f.write(f"\nCAUTION - MODERATE IMBALANCE: Monitor training\n")
            else:
                f.write(f"\nOK - GOOD BALANCE: Dataset is well balanced\n")
    
    # Print to console
    print("\n" + "=" * 80)
    print("CLEANED DATASET CLASS DISTRIBUTION")
    print("=" * 80)
    print("\nSPLIT SUMMARY:")
    total_images = 0
    total_annotations = 0
    for split in ['train', 'val', 'test']:
        if split in split_stats:
            stats = split_stats[split]
            print(f"{split.upper():10s}: {stats['images']:3d} images, {stats['annotations']:4d} annotations")
            total_images += stats['images']
            total_annotations += stats['annotations']
    print(f"\nTOTAL     : {total_images:3d} images, {total_annotations:4d} annotations")
    
    print("\nCLASS DISTRIBUTION:")
    print("-" * 80)
    
    for class_name in classes:
        stats = class_stats[class_name]
        count = stats['annotations']
        images = len(stats['images'])
        print(f"{class_name:<20} {count:6d} annotations in {images:3d} images")
    
    print(f"\n[OK] Report saved: {report_path}")

if __name__ == "__main__":
    analyze_cleaned_dataset()
