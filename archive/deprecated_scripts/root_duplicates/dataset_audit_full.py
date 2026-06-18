"""
Comprehensive Dataset Audit for YOLOv8
Checks all data quality issues
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

class DatasetAuditor:
    def __init__(self, dataset_path, classes_file):
        self.dataset_path = Path(dataset_path)
        self.classes = self._load_classes(classes_file)
        self.num_classes = len(self.classes)
        self.issues = {
            'missing_labels': [],
            'empty_labels': [],
            'corrupted_labels': [],
            'invalid_class_ids': [],
            'invalid_bbox': [],
            'tiny_bbox': [],
            'oversized_bbox': [],
            'corrupted_images': [],
            'duplicate_labels': [],
        }
        self.class_stats = defaultdict(lambda: {'count': 0, 'images': set()})
        self.bbox_stats = []
        
    def _load_classes(self, classes_file):
        """Load class names from classes.txt"""
        classes = []
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    
    def audit_dataset(self):
        """Run complete dataset audit"""
        print("=" * 80)
        print("STARTING COMPREHENSIVE DATASET AUDIT")
        print("=" * 80)
        
        # Audit each split
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists():
                print(f"\n⚠️  {split.upper()} split not found, skipping...")
                continue
                
            print(f"\n{'='*80}")
            print(f"AUDITING {split.upper()} SPLIT")
            print(f"{'='*80}")
            
            self._audit_split(split, images_dir, labels_dir)
        
        # Generate report
        self._generate_report()
        
        return self.issues, self.class_stats, self.bbox_stats
    
    def _audit_split(self, split, images_dir, labels_dir):
        """Audit a single split"""
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nTotal images found: {len(image_files)}")
        
        for img_file in image_files:
            img_path = images_dir / img_file
            label_file = labels_dir / (img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            # Check image
            self._check_image(img_path, img_file, split)
            
            # Check label
            if not label_file.exists():
                self.issues['missing_labels'].append(str(img_path))
            else:
                self._check_label(label_file, img_path, img_file, split)
    
    def _check_image(self, img_path, img_file, split):
        """Check image validity"""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                self.issues['corrupted_images'].append(f"{split}/{img_file}")
                return
            
            if img.size == 0:
                self.issues['corrupted_images'].append(f"{split}/{img_file}")
                return
                
            img_height, img_width = img.shape[:2]
            return img_width, img_height
        except Exception as e:
            self.issues['corrupted_images'].append(f"{split}/{img_file}: {str(e)}")
            return None
    
    def _check_label(self, label_file, img_path, img_file, split):
        """Check label validity"""
        try:
            # Get image dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                return
            img_height, img_width = img.shape[:2]
            
            # Read label
            with open(label_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                self.issues['empty_labels'].append(f"{split}/{img_file}")
                return
            
            # Check each annotation
            seen_annotations = set()
            for line in lines:
                try:
                    parts = line.split()
                    
                    if len(parts) < 5:
                        self.issues['corrupted_labels'].append(f"{split}/{img_file}: Invalid format")
                        continue
                    
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check class ID
                    if class_id < 0 or class_id >= self.num_classes:
                        self.issues['invalid_class_ids'].append(
                            f"{split}/{img_file}: Invalid class {class_id} (valid: 0-{self.num_classes-1})"
                        )
                        continue
                    
                    # Check normalized coords are in [0,1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        self.issues['invalid_bbox'].append(
                            f"{split}/{img_file}: Coords out of range ({x_center}, {y_center}, {width}, {height})"
                        )
                        continue
                    
                    # Convert to pixel coords
                    px_width = width * img_width
                    px_height = height * img_height
                    px_area = px_width * px_height
                    
                    # Check bbox size
                    if px_area < 16:  # Less than 4x4 pixels
                        self.issues['tiny_bbox'].append(
                            f"{split}/{img_file}: Tiny bbox {px_width:.1f}x{px_height:.1f}px ({px_area:.0f}px²)"
                        )
                    
                    if px_area > (img_width * img_height * 0.95):  # 95% of image
                        self.issues['oversized_bbox'].append(
                            f"{split}/{img_file}: Oversized bbox {px_width:.1f}x{px_height:.1f}px ({px_area:.0f}px²)"
                        )
                    
                    # Track class stats
                    annotation_key = f"{class_id}:{x_center:.4f}:{y_center:.4f}:{width:.4f}:{height:.4f}"
                    if annotation_key in seen_annotations:
                        self.issues['duplicate_labels'].append(
                            f"{split}/{img_file}: Duplicate annotation"
                        )
                    seen_annotations.add(annotation_key)
                    
                    self.class_stats[self.classes[class_id]]['count'] += 1
                    self.class_stats[self.classes[class_id]]['images'].add(img_file)
                    
                    self.bbox_stats.append({
                        'class': self.classes[class_id],
                        'width': px_width,
                        'height': px_height,
                        'area': px_area
                    })
                    
                except ValueError as e:
                    self.issues['corrupted_labels'].append(f"{split}/{img_file}: {str(e)}")
                    
        except Exception as e:
            self.issues['corrupted_labels'].append(f"{split}/{img_file}: {str(e)}")
    
    def _generate_report(self):
        """Generate comprehensive audit report"""
        report_path = self.dataset_path.parent / 'dataset_audit_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE DATASET AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            total_issues = sum(len(v) for v in self.issues.values())
            f.write(f"TOTAL ISSUES FOUND: {total_issues}\n\n")
            
            # Issues by category
            f.write("ISSUES BY CATEGORY:\n")
            f.write("-" * 80 + "\n")
            
            for issue_type, items in self.issues.items():
                if items:
                    f.write(f"\n{issue_type.upper()}: {len(items)} found\n")
                    for item in items[:10]:  # Show first 10
                        f.write(f"  - {item}\n")
                    if len(items) > 10:
                        f.write(f"  ... and {len(items) - 10} more\n")
            
            # Class distribution
            f.write("\n" + "=" * 80 + "\n")
            f.write("CLASS DISTRIBUTION:\n")
            f.write("-" * 80 + "\n")
            
            for class_name in self.classes:
                count = self.class_stats[class_name]['count']
                images = len(self.class_stats[class_name]['images'])
                f.write(f"{class_name:20s}: {count:6d} annotations in {images:4d} images\n")
            
            # Bounding box statistics
            if self.bbox_stats:
                f.write("\n" + "=" * 80 + "\n")
                f.write("BOUNDING BOX STATISTICS:\n")
                f.write("-" * 80 + "\n")
                
                areas = [s['area'] for s in self.bbox_stats]
                widths = [s['width'] for s in self.bbox_stats]
                heights = [s['height'] for s in self.bbox_stats]
                
                f.write(f"Total bboxes: {len(self.bbox_stats)}\n")
                f.write(f"Bbox areas - Min: {min(areas):.0f}px, Max: {max(areas):.0f}px, Avg: {np.mean(areas):.0f}px\n")
                f.write(f"Bbox widths - Min: {min(widths):.0f}px, Max: {max(widths):.0f}px, Avg: {np.mean(widths):.0f}px\n")
                f.write(f"Bbox heights - Min: {min(heights):.0f}px, Max: {max(heights):.0f}px, Avg: {np.mean(heights):.0f}px\n")
            
            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")
            
            recommendations = []
            
            if self.issues['missing_labels']:
                recommendations.append(f"- Remove {len(self.issues['missing_labels'])} images without labels")
            
            if self.issues['corrupted_labels']:
                recommendations.append(f"- Fix or remove {len(self.issues['corrupted_labels'])} corrupted labels")
            
            if self.issues['invalid_class_ids']:
                recommendations.append(f"- Fix {len(self.issues['invalid_class_ids'])} invalid class IDs")
            
            if self.issues['invalid_bbox']:
                recommendations.append(f"- Fix {len(self.issues['invalid_bbox'])} invalid bounding boxes")
            
            if self.issues['tiny_bbox']:
                recommendations.append(f"- Remove or fix {len(self.issues['tiny_bbox'])} tiny bounding boxes")
            
            if self.issues['oversized_bbox']:
                recommendations.append(f"- Fix {len(self.issues['oversized_bbox'])} oversized bounding boxes")
            
            # Check class imbalance
            if self.class_stats:
                counts = [self.class_stats[c]['count'] for c in self.classes]
                if counts:
                    max_count = max(counts)
                    min_count = min(counts)
                    if max_count > 0 and min_count > 0:
                        imbalance_ratio = max_count / min_count
                        if imbalance_ratio > 5:
                            recommendations.append(f"- Address class imbalance (ratio: {imbalance_ratio:.1f}:1)")
            
            for rec in recommendations:
                f.write(rec + "\n")
        
        print(f"\n✅ Audit report saved: {report_path}")
        
        # Also print to console
        print("\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)
        
        for issue_type, items in self.issues.items():
            if items:
                print(f"\n⚠️  {issue_type}: {len(items)} found")
        
        print("\n" + "=" * 80)
        print("CLASS DISTRIBUTION")
        print("=" * 80)
        
        for class_name in self.classes:
            count = self.class_stats[class_name]['count']
            images = len(self.class_stats[class_name]['images'])
            print(f"{class_name:20s}: {count:6d} annotations in {images:4d} images")


if __name__ == "__main__":
    dataset_path = r"C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase\dataset_final"
    classes_file = os.path.join(dataset_path, "classes.txt")
    
    auditor = DatasetAuditor(dataset_path, classes_file)
    issues, class_stats, bbox_stats = auditor.audit_dataset()
    
    print("\n✅ Dataset audit completed!")
