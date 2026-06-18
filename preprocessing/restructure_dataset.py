"""
Restructure the YOLO dataset to have only damage types as YOLO classes,
with location and severity stored as metadata.

Old class mapping (0-indexed):
0: dent          (damage)
1: scratch       (damage)
2: crack         (damage)
3: broken_part   (damage)
4: paint_damage  (damage)
5: other_damage  (damage)
6: front_bumper  (location)
7: rear_bumper   (location)
8: hood          (location)
9: windshield    (location)
10: left_door    (location)
11: right_door   (location)
12: roof         (location)
13: side_panel   (location)
14: other_location (location)
15: low           (severity)
16: medium        (severity)
17: high          (severity)

New class mapping (0-indexed):
0: dent
1: scratch
2: crack
3: broken_part
4: paint_damage
5: other_damage
"""

import os
import csv
import shutil
from collections import Counter, defaultdict

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "datasets", "dataset_final")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "new_dataset")

# Class definitions
DAMAGE_CLASSES = [
    "dent",
    "scratch",
    "crack",
    "broken_part",
    "paint_damage",
    "other_damage"
]

LOCATION_CLASSES = [
    "front_bumper",
    "rear_bumper",
    "hood",
    "windshield",
    "left_door",
    "right_door",
    "roof",
    "side_panel",
    "other_location"
]

SEVERITY_CLASSES = ["low", "medium", "high"]

# Old class ID mapping (0-indexed) -> class name
OLD_CLASSES = DAMAGE_CLASSES + LOCATION_CLASSES + SEVERITY_CLASSES
OLD_ID_TO_NAME = {i: name for i, name in enumerate(OLD_CLASSES)}
OLD_NAME_TO_ID = {name: i for i, name in enumerate(OLD_CLASSES)}

# New class mapping (0-indexed) -> class name
NEW_CLASSES = DAMAGE_CLASSES
NEW_ID_TO_NAME = {i: name for i, name in enumerate(NEW_CLASSES)}
NEW_NAME_TO_ID = {name: i for i, name in enumerate(NEW_CLASSES)}

# Old IDs for each category
DAMAGE_OLD_IDS = set(range(0, 6))       # 0-5
LOCATION_OLD_IDS = set(range(6, 15))    # 6-14
SEVERITY_OLD_IDS = set(range(15, 18))   # 15-17

print("=" * 60)
print("DATASET RESTRUCTURING SCRIPT")
print("=" * 60)
print()
print(f"Source: {SOURCE_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()
print(f"Damage types (YOLO classes): {DAMAGE_CLASSES}")
print(f"Locations (metadata): {LOCATION_CLASSES}")
print(f"Severities (metadata): {SEVERITY_CLASSES}")
print()

# --- Step 1: Prepare output directories ---
print("Step 1: Creating output directory structure...")
os.makedirs(os.path.join(OUTPUT_DIR, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "val"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels", "test"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reports"), exist_ok=True)

# --- Step 2: Write new classes.txt ---
print("Step 2: Writing new classes.txt...")
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for cls in NEW_CLASSES:
        f.write(cls + "\n")

# --- Step 3: Process all splits ---
print("Step 3: Processing labels and generating metadata...")

metadata_rows = []
damage_type_counter = Counter()
location_counter = Counter()
severity_counter = Counter()
total_annotations = 0
total_images_with_annotations = 0
errors = []
warnings = []
images_copied = 0
images_not_found = []

# Track matching images for each label
splits = ["train", "val", "test"]
image_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]

for split in splits:
    labels_dir = os.path.join(SOURCE_DIR, "labels", split)
    images_src_dir = os.path.join(SOURCE_DIR, "images", split)
    
    if not os.path.isdir(labels_dir):
        print(f"  Labels directory not found: {labels_dir}")
        continue
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt") and not f.endswith(".cache")]
    
    for label_file in sorted(label_files):
        label_path = os.path.join(labels_dir, label_file)
        stem = os.path.splitext(label_file)[0]
        
        # Find matching image
        image_path_src = None
        for ext in image_extensions:
            candidate = os.path.join(images_src_dir, stem + ext)
            if os.path.isfile(candidate):
                image_path_src = candidate
                break
        
        if image_path_src is None:
            errors.append(f"  [ERROR] No image found for label: {label_file} in {split}")
            continue
        
        # Read label file
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        # Group annotations by bounding box coordinates (3 lines per box: damage, location, severity)
        # Each triplet has identical bbox coords
        annotations = []  # list of dicts
        current_box = None
        current_damage = None
        current_location = None
        current_severity = None
        bbox_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                warnings.append(f"  Malformed line in {label_file}: {line}")
                continue
            
            class_id = int(parts[0])
            coords = " ".join(parts[1:5])
            
            if class_id in DAMAGE_OLD_IDS:
                if current_damage is not None:
                    # New annotation starting, save previous
                    if current_damage is not None and current_location is not None and current_severity is not None:
                        annotations.append({
                            'damage_old_id': current_damage,
                            'location_old_id': current_location,
                            'severity_old_id': current_severity,
                            'bbox': current_box
                        })
                current_damage = class_id
                current_box = coords
                current_location = None
                current_severity = None
            elif class_id in LOCATION_OLD_IDS:
                current_location = class_id
            elif class_id in SEVERITY_OLD_IDS:
                current_severity = class_id
            else:
                warnings.append(f"  Unknown class ID {class_id} in {label_file}")
        
        # Save last annotation
        if current_damage is not None and current_location is not None and current_severity is not None:
            annotations.append({
                'damage_old_id': current_damage,
                'location_old_id': current_location,
                'severity_old_id': current_severity,
                'bbox': current_box
            })
        
        if not annotations:
            warnings.append(f"  No valid annotations found in {label_file}")
            continue
        
        # Write new label file (only damage types with remapped class IDs)
        new_label_path = os.path.join(OUTPUT_DIR, "labels", split, label_file)
        with open(new_label_path, "w") as f:
            for ann in annotations:
                new_class_id = ann['damage_old_id']  # damage types map 1:1 (0-5 remain 0-5)
                f.write(f"{new_class_id} {ann['bbox']}\n")
        
        # Copy image
        image_ext = os.path.splitext(image_path_src)[1]
        image_name = stem + image_ext
        image_path_dst = os.path.join(OUTPUT_DIR, "images", split, image_name)
        
        # Handle case where image already exists at destination
        if os.path.isfile(image_path_dst):
            # Only copy if different
            if os.path.getsize(image_path_src) != os.path.getsize(image_path_dst):
                shutil.copy2(image_path_src, image_path_dst)
        else:
            shutil.copy2(image_path_src, image_path_dst)
        images_copied += 1
        
        # Build metadata rows for this image
        for ann in annotations:
            damage_name = OLD_ID_TO_NAME[ann['damage_old_id']]
            location_name = OLD_ID_TO_NAME[ann['location_old_id']]
            severity_name = OLD_ID_TO_NAME[ann['severity_old_id']]
            bbox = ann['bbox']
            
            metadata_rows.append({
                'image_name': image_name,
                'damage_type': damage_name,
                'location': location_name,
                'severity': severity_name,
                'bbox_x1': bbox.split()[0],
                'bbox_y1': bbox.split()[1],
                'bbox_x2': bbox.split()[2],
                'bbox_y2': bbox.split()[3]
            })
            
            total_annotations += 1
            damage_type_counter[damage_name] += 1
            location_counter[location_name] += 1
            severity_counter[severity_name] += 1
        
        total_images_with_annotations += 1

print()
print(f"  Processed {total_images_with_annotations} images")
print(f"  Total annotations: {total_annotations}")
print(f"  Images copied: {images_copied}")

# --- Step 4: Write metadata CSV ---
print("Step 4: Writing damage_metadata.csv...")
csv_path = os.path.join(OUTPUT_DIR, "damage_metadata.csv")
with open(csv_path, "w", newline="") as f:
    fieldnames = ["image_name", "damage_type", "location", "severity", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in metadata_rows:
        writer.writerow(row)

# --- Step 5: Write data.yaml ---
print("Step 5: Writing data.yaml...")
yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {os.path.abspath(OUTPUT_DIR).replace(chr(92), '/')}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write(f"nc: {len(NEW_CLASSES)}\n")
    f.write("names:\n")
    for cls in NEW_CLASSES:
        f.write(f"  - {cls}\n")

# --- Step 6: Validation ---
print("Step 6: Running validation...")
validation_passed = True
validation_issues = []

# 6.1: Every image has matching label file
print("  6.1: Checking image-label correspondence...")
for split in splits:
    images_out_dir = os.path.join(OUTPUT_DIR, "images", split)
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    
    if not os.path.isdir(images_out_dir):
        continue
    
    # Check labels exist for each image
    for img_file in sorted(os.listdir(images_out_dir)):
        if img_file.startswith("."):
            continue
        stem = os.path.splitext(img_file)[0]
        label_file = stem + ".txt"
        label_path = os.path.join(labels_out_dir, label_file)
        if not os.path.isfile(label_path):
            validation_issues.append(f"  Missing label for image: {split}/{img_file}")
            validation_passed = False

# 6.2: Every label class exists in classes.txt (only damage types 0-5)
print("  6.2: Checking label class IDs are valid...")
damage_ids = set(range(len(NEW_CLASSES)))
for split in splits:
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    if not os.path.isdir(labels_out_dir):
        continue
    for label_file in sorted(os.listdir(labels_out_dir)):
        if not label_file.endswith(".txt") or label_file.endswith(".cache"):
            continue
        label_path = os.path.join(labels_out_dir, label_file)
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_id = int(parts[0])
                    if cls_id not in damage_ids:
                        validation_issues.append(f"  Invalid class ID {cls_id} in {split}/{label_file}")
                        validation_passed = False
                except (ValueError, IndexError):
                    validation_issues.append(f"  Malformed line in {split}/{label_file}: {line}")
                    validation_passed = False

# 6.3: No location or severity classes remain
print("  6.3: Checking no location/severity classes...")
location_severity_old_ids = LOCATION_OLD_IDS | SEVERITY_OLD_IDS
for split in splits:
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    if not os.path.isdir(labels_out_dir):
        continue
    for label_file in sorted(os.listdir(labels_out_dir)):
        if not label_file.endswith(".txt") or label_file.endswith(".cache"):
            continue
        label_path = os.path.join(labels_out_dir, label_file)
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    cls_id = int(parts[0])
                    if cls_id in location_severity_old_ids:
                        validation_issues.append(f"  Location/severity class {cls_id} found in {split}/{label_file}")
                        validation_passed = False
                except (ValueError, IndexError):
                    pass

# 6.4: Verify metadata CSV matches annotations
print("  6.4: Checking metadata CSV integrity...")
image_damage_pairs_metadata = set()
for row in metadata_rows:
    image_damage_pairs_metadata.add((row['image_name'], row['damage_type']))

for split in splits:
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    if not os.path.isdir(labels_out_dir):
        continue
    for label_file in sorted(os.listdir(labels_out_dir)):
        if not label_file.endswith(".txt") or label_file.endswith(".cache"):
            continue
        stem = os.path.splitext(label_file)[0]
        label_path = os.path.join(labels_out_dir, label_file)
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                damage_name = NEW_ID_TO_NAME[int(parts[0])]
                # Find matching image
                for ext in image_extensions:
                    candidate = os.path.join(OUTPUT_DIR, "images", split, stem + ext)
                    if os.path.isfile(candidate):
                        img_name = stem + ext
                        break
                else:
                    continue
                
                # Check this damage type exists in metadata for this image
                found = False
                for row in metadata_rows:
                    if row['image_name'] == img_name and row['damage_type'] == damage_name:
                        found = True
                        break
                if not found:
                    validation_issues.append(f"  Metadata missing: {img_name} / {damage_name}")

# Write validation report
print("  6.5: Writing validation report...")
report_path = os.path.join(OUTPUT_DIR, "reports", "validation_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("VALIDATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    
    if validation_passed:
        f.write("Status: PASSED\n")
    else:
        f.write("Status: FAILED\n")
    
    f.write(f"\nValidation Checks:\n")
    f.write(f"  Image-label correspondence: {'PASS' if validation_passed else 'FAIL'}\n")
    f.write(f"  Valid class IDs only: {'PASS' if validation_passed else 'See issues'}\n")
    f.write(f"  No location/severity in YOLO: {'PASS' if validation_passed else 'See issues'}\n")
    f.write(f"  Metadata integrity: {'PASS' if validation_passed else 'See issues'}\n")
    
    if validation_issues:
        f.write(f"\nIssues Found ({len(validation_issues)}):\n")
        for issue in validation_issues:
            f.write(f"  {issue}\n")
    
    if not validation_passed:
        f.write(f"\nWARNING: Validation did not pass. Please review issues above.\n")

# --- Step 7: Generate summary ---
print("Step 7: Generating summary...")
summary_path = os.path.join(OUTPUT_DIR, "reports", "dataset_summary.txt")
total_images = 0
for split in splits:
    img_dir = os.path.join(OUTPUT_DIR, "images", split)
    if os.path.isdir(img_dir):
        total_images += len([f for f in os.listdir(img_dir) if not f.startswith(".")])

with open(summary_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("DATASET SUMMARY - new_dataset\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Total Images: {total_images}\n")
    f.write(f"Total Annotations: {total_annotations}\n")
    f.write(f"Images with Annotations: {total_images_with_annotations}\n")
    f.write(f"YOLO Classes (Damage Types): {len(NEW_CLASSES)}\n\n")
    
    f.write("Damage Type Distribution:\n")
    f.write("-" * 40 + "\n")
    for cls_name in NEW_CLASSES:
        count = damage_type_counter.get(cls_name, 0)
        pct = count / total_annotations * 100 if total_annotations > 0 else 0
        f.write(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)\n")
    f.write(f"  {'TOTAL':25s}: {total_annotations:5d}\n\n")
    
    f.write("Location Distribution:\n")
    f.write("-" * 40 + "\n")
    loc_total = sum(location_counter.values())
    for cls_name in LOCATION_CLASSES:
        count = location_counter.get(cls_name, 0)
        pct = count / loc_total * 100 if loc_total > 0 else 0
        f.write(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)\n")
    f.write(f"  {'TOTAL':25s}: {loc_total:5d}\n\n")
    
    f.write("Severity Distribution:\n")
    f.write("-" * 40 + "\n")
    sev_total = sum(severity_counter.values())
    for cls_name in SEVERITY_CLASSES:
        count = severity_counter.get(cls_name, 0)
        pct = count / sev_total * 100 if sev_total > 0 else 0
        f.write(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)\n")
    f.write(f"  {'TOTAL':25s}: {sev_total:5d}\n\n")
    
    f.write("Split Distribution:\n")
    f.write("-" * 40 + "\n")
    for split in splits:
        img_dir = os.path.join(OUTPUT_DIR, "images", split)
        lbl_dir = os.path.join(OUTPUT_DIR, "labels", split)
        img_count = len([f for f in os.listdir(img_dir) if not f.startswith(".")]) if os.path.isdir(img_dir) else 0
        lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith(".txt") and not f.endswith(".cache")]) if os.path.isdir(lbl_dir) else 0
        f.write(f"  {split:10s}: {img_count:4d} images, {lbl_count:4d} labels\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("End of Summary\n")
    f.write("=" * 60 + "\n")

# Also print to console
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total Images: {total_images}")
print(f"Total Annotations: {total_annotations}")
print(f"YOLO Classes (Damage Types): {len(NEW_CLASSES)}")
print()
print("Damage Type Distribution:")
for cls_name in NEW_CLASSES:
    count = damage_type_counter.get(cls_name, 0)
    pct = count / total_annotations * 100 if total_annotations > 0 else 0
    print(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)")
print()
print("Location Distribution:")
for cls_name in LOCATION_CLASSES:
    count = location_counter.get(cls_name, 0)
    pct = count / loc_total * 100 if loc_total > 0 else 0
    print(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)")
print()
print("Severity Distribution:")
for cls_name in SEVERITY_CLASSES:
    count = severity_counter.get(cls_name, 0)
    pct = count / sev_total * 100 if sev_total > 0 else 0
    print(f"  {cls_name:25s}: {count:5d} ({pct:5.1f}%)")
print()
print(f"Validation: {'PASSED' if validation_passed else 'FAILED - check validation report'}")
print(f"\nOutput saved to: {OUTPUT_DIR}")
print(f"  - images/ (train/val/test)")
print(f"  - labels/ (train/val/test)")
print(f"  - classes.txt")
print(f"  - data.yaml")
print(f"  - damage_metadata.csv")
print(f"  - reports/validation_report.txt")
print(f"  - reports/dataset_summary.txt")
print()

if errors:
    print(f"Errors ({len(errors)}):")
    for e in errors:
        print(f"  {e}")

if warnings:
    print(f"Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  {w}")
