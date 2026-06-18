"""
Clean and restructure dataset for proper 5-class damage detection
- Remove empty labels
- Filter to only the 5 target damage classes
- Balance dataset
- Create proper YAML config
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import yaml

class DatasetCleaner:
    # Map ANY annotation to the 5 target classes
    TARGET_CLASSES = ['dent', 'scratch', 'crack', 'broken_part', 'paint_damage']
    
    # Keep location and severity info - don't filter on them
    # They may contain damage of our target types
    
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = Path(source_dataset)
        self.target_dataset = Path(target_dataset)
        self.source_classes = self._load_classes()
        self.stats = {'removed': 0, 'kept': 0, 'filtered_annotations': 0}
        
    def _load_classes(self):
        """Load source class names"""
        classes_file = self.source_dataset / "classes.txt"
        classes = []
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    
    def clean(self):
        """Clean entire dataset"""
        print("=" * 80)
        print("CLEANING DATASET FOR 5-CLASS DAMAGE DETECTION")
        print("=" * 80)
        
        # Create target directories
        self.target_dataset.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.target_dataset / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.target_dataset / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*80}")
            print(f"Processing {split.upper()} split")
            print(f"{'='*80}")
            self._process_split(split)
        
        # Create new YAML config
        self._create_yaml_config()
        
        # Create new classes file
        self._create_classes_file()
        
        # Print summary
        self._print_summary()
    
    def _process_split(self, split):
        """Process a single split"""
        src_images_dir = self.source_dataset / 'images' / split
        src_labels_dir = self.source_dataset / 'labels' / split
        dst_images_dir = self.target_dataset / 'images' / split
        dst_labels_dir = self.target_dataset / 'labels' / split
        
        if not src_images_dir.exists():
            print(f"⚠️  {split} split not found")
            return
        
        image_files = sorted([f for f in os.listdir(src_images_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        processed = 0
        kept = 0
        removed_empty = 0
        
        for img_file in image_files:
            src_img_path = src_images_dir / img_file
            label_file = src_labels_dir / (img_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
            
            processed += 1
            
            # Check if label exists
            if not label_file.exists():
                print(f"  ❌ {img_file}: No label file")
                self.stats['removed'] += 1
                continue
            
            # Read and filter annotations
            filtered_annotations = self._filter_annotations(label_file)
            
            # If no valid annotations after filtering, skip
            if not filtered_annotations:
                removed_empty += 1
                print(f"  ❌ {img_file}: Empty after filtering")
                self.stats['removed'] += 1
                continue
            
            # Copy image
            dst_img_path = dst_images_dir / img_file
            shutil.copy2(src_img_path, dst_img_path)
            
            # Write filtered labels
            dst_label_path = dst_labels_dir / label_file.name
            with open(dst_label_path, 'w') as f:
                for ann in filtered_annotations:
                    f.write(ann + '\n')
            
            kept += 1
            self.stats['kept'] += 1
            print(f"  ✅ {img_file}: Kept with {len(filtered_annotations)} annotations")
        
        print(f"\n{split.upper()} Summary:")
        print(f"  Processed: {processed}")
        print(f"  Kept: {kept}")
        print(f"  Removed: {processed - kept}")
        print(f"  Removed (empty): {removed_empty}")
    
    def _filter_annotations(self, label_file):
        """Filter annotations to keep only target classes"""
        filtered = []
        
        try:
            with open(label_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            for line in lines:
                try:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Check if class_id is valid
                    if class_id < 0 or class_id >= len(self.source_classes):
                        continue
                    
                    class_name = self.source_classes[class_id]
                    
                    # Keep only target damage classes
                    if class_name in self.TARGET_CLASSES:
                        # Remap class ID to new 5-class scheme
                        new_class_id = self.TARGET_CLASSES.index(class_name)
                        new_line = f"{new_class_id} " + " ".join(parts[1:])
                        filtered.append(new_line)
                        self.stats['filtered_annotations'] += 1
                    
                except (ValueError, IndexError):
                    continue
        
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
        
        return filtered
    
    def _create_yaml_config(self):
        """Create new YAML config with 5 classes"""
        yaml_content = {
            'path': str(self.target_dataset),
            'train': str(self.target_dataset / 'images' / 'train'),
            'val': str(self.target_dataset / 'images' / 'val'),
            'test': str(self.target_dataset / 'images' / 'test'),
            'nc': len(self.TARGET_CLASSES),
            'names': self.TARGET_CLASSES
        }
        
        yaml_file = self.target_dataset / 'data.yaml'
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Created YAML config: {yaml_file}")
    
    def _create_classes_file(self):
        """Create new classes.txt file"""
        classes_file = self.target_dataset / 'classes.txt'
        with open(classes_file, 'w') as f:
            for cls in self.TARGET_CLASSES:
                f.write(cls + '\n')
        
        print(f"✅ Created classes file: {classes_file}")
    
    def _print_summary(self):
        """Print cleaning summary"""
        print("\n" + "=" * 80)
        print("DATASET CLEANING SUMMARY")
        print("=" * 80)
        print(f"Images kept: {self.stats['kept']}")
        print(f"Images removed: {self.stats['removed']}")
        print(f"Annotations filtered: {self.stats['filtered_annotations']}")
        print(f"\nTarget classes: {', '.join(self.TARGET_CLASSES)}")
        print(f"New class count: {len(self.TARGET_CLASSES)}")
        print(f"\nCleaned dataset: {self.target_dataset}")


if __name__ == "__main__":
    source = "datasets/dataset_final"
    target = "datasets/dataset_cleaned"
    
    cleaner = DatasetCleaner(source, target)
    cleaner.clean()
