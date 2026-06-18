#!/usr/bin/env python3
"""
Complete Dataset Labeling Pipeline for Vehicle Damage Detection
Converts binary classification dataset into detailed defect dataset for YOLOv8 object detection.

Generates:
- Bounding boxes for damaged regions
- Damage type, location, and severity annotations
- YOLO format labels
- Data.yaml for YOLOv8
- Annotation reports and statistics
- Flags uncertain images for manual review
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import yaml
import shutil
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from skimage import measure, morphology, feature
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / 'datasets' / 'Dataset' / 'data1a'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'dataset_labeled'

# Input paths
TRAIN_DAMAGE = DATASET_DIR / 'training' / '00-damage'
TRAIN_WHOLE = DATASET_DIR / 'training' / '01-whole'
VAL_DAMAGE = DATASET_DIR / 'validation' / '00-damage'
VAL_WHOLE = DATASET_DIR / 'validation' / '01-whole'

# Output paths
OUT_IMAGES_TRAIN = OUTPUT_DIR / 'images' / 'train'
OUT_IMAGES_VAL = OUTPUT_DIR / 'images' / 'val'
OUT_IMAGES_TEST = OUTPUT_DIR / 'images' / 'test'
OUT_LABELS_TRAIN = OUTPUT_DIR / 'labels' / 'train'
OUT_LABELS_VAL = OUTPUT_DIR / 'labels' / 'val'
OUT_LABELS_TEST = OUTPUT_DIR / 'labels' / 'test'

# YOLO class mapping
CLASS_NAMES = [
    'dent',           # 0
    'scratch',        # 1
    'crack',          # 2
    'broken_part',    # 3
    'paint_damage',   # 4
    'other_damage',   # 5
]

LOCATION_NAMES = [
    'front_bumper',   # 6
    'rear_bumper',    # 7
    'hood',           # 8
    'windshield',     # 9
    'left_door',      # 10
    'right_door',     # 11
    'roof',           # 12
    'side_panel',     # 13
    'other_location', # 14
]

SEVERITY_NAMES = [
    'low',    # 15
    'medium', # 16
    'high',   # 17
]

# Combined: 6 damage types + 9 locations + 3 severities = 18 classes
ALL_CLASSES = CLASS_NAMES + LOCATION_NAMES + SEVERITY_NAMES
NUM_CLASSES = len(ALL_CLASSES)

# Confidence threshold: if max confidence < this, flag for review
CONFIDENCE_THRESHOLD = 0.6

# Manual review folder
MANUAL_REVIEW_DIR = OUTPUT_DIR / 'manual_review'

# Image preprocessing
TARGET_SIZE = 640  # YOLOv8 default
CONFIDENCE_MAP = {}

def setup_directories():
    """Create all necessary output directories."""
    dirs = [
        OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_IMAGES_TEST,
        OUT_LABELS_TRAIN, OUT_LABELS_VAL, OUT_LABELS_TEST,
        MANUAL_REVIEW_DIR,
        OUTPUT_DIR / 'reports',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directories created under {OUTPUT_DIR}")


def load_image(image_path):
    """Load and preprocess an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    orig = img.copy()
    h, w = img.shape[:2]
    # Resize maintaining aspect ratio
    scale = min(TARGET_SIZE / w, TARGET_SIZE / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    # Pad to square
    delta_w = TARGET_SIZE - new_w
    delta_h = TARGET_SIZE - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, (w, h, orig)


def get_image_region_mask(h, w, region_name):
    """
    Estimate bounding box region for a given vehicle part.
    Returns (x1, y1, x2, y2) in relative coordinates (0-1).
    """
    if region_name == 'front_bumper':
        return (0.1, 0.8, 0.9, 0.99)
    elif region_name == 'rear_bumper':
        return (0.1, 0.01, 0.9, 0.2)
    elif region_name == 'hood':
        return (0.15, 0.2, 0.85, 0.5)
    elif region_name == 'windshield':
        return (0.2, 0.15, 0.8, 0.6)
    elif region_name == 'left_door':
        return (0.0, 0.25, 0.35, 0.75)
    elif region_name == 'right_door':
        return (0.65, 0.25, 1.0, 0.75)
    elif region_name == 'roof':
        return (0.1, 0.0, 0.9, 0.25)
    elif region_name == 'side_panel':
        return (0.3, 0.2, 0.7, 0.8)
    else:  # other_location
        return (0.0, 0.0, 1.0, 1.0)


def detect_damage_contours(image):
    """
    Detect damage regions using computer vision techniques.
    Returns list of (bbox, confidence, features) for each detected region.
    """
    h, w = image.shape[:2]
    results = []

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Edge detection (Canny)
    edges = cv2.Canny(blurred, 50, 150)

    # 4. Morphological operations to close gaps
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)

    # 5. Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # 6. Analyze each contour
    valid_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > w * h * 0.8:  # Filter too small/large
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / bh if bh > 0 else 0

        # Compute region confidence based on features
        region = image[y:y+bh, x:x+bw]
        if region.size == 0:
            continue

        # Texture analysis using Laplacian variance
        laplacian_var = cv2.Laplacian(gray[y:y+bh, x:x+bw], cv2.CV_64F).var()

        # Color variance analysis
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:, :, 1])  # Saturation std

        # Intensity difference from surroundings
        region_mean = np.mean(gray[y:y+bh, x:x+bw])
        # Surrounding region (dilated)
        y1_s = max(0, y-10)
        y2_s = min(h, y+bh+10)
        x1_s = max(0, x-10)
        x2_s = min(w, x+bw+10)
        surround_mean = np.mean(gray[y1_s:y2_s, x1_s:x2_s])
        intensity_diff = abs(float(region_mean) - float(surround_mean))

        # Calculate confidence score
        # High Laplacian variance = more texture (damage indicator)
        # High color std = more color variation (damage indicator)  
        # High intensity diff = more contrast with surroundings
        edge_density = area / (bw * bh) if bw * bh > 0 else 0

        confidence = min(1.0, (
            0.3 * min(laplacian_var / 500, 1.0) +
            0.3 * min(color_std / 80, 1.0) +
            0.2 * min(intensity_diff / 50, 1.0) +
            0.2 * edge_density
        ))

        valid_regions.append({
            'bbox': (x, y, bw, bh),
            'confidence': confidence,
            'contour': cnt,
            'area': area,
            'laplacian_var': laplacian_var,
            'color_std': color_std,
            'intensity_diff': intensity_diff,
        })

    # Remove overlapping regions (keep highest confidence)
    valid_regions.sort(key=lambda r: r['confidence'], reverse=True)
    filtered = []
    for region in valid_regions:
        x1, y1, bw, bh = region['bbox']
        x2, y2 = x1 + bw, y1 + bh
        overlapping = False
        for kept in filtered:
            kx1, ky1, kbw, kbh = kept['bbox']
            kx2, ky2 = kx1 + kbw, ky1 + kbh
            # IoU calculation
            inter_x1 = max(x1, kx1)
            inter_y1 = max(y1, ky1)
            inter_x2 = min(x2, kx2)
            inter_y2 = min(y2, ky2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area1 = bw * bh
            area2 = kbw * kbh
            iou = inter_area / (area1 + area2 - inter_area) if (area1 + area2 - inter_area) > 0 else 0
            if iou > 0.5:
                overlapping = True
                break
        if not overlapping:
            filtered.append(region)

    return filtered


def estimate_damage_type(region, image):
    """
    Estimate damage type based on visual features.
    Returns (damage_type, confidence).
    """
    x, y, bw, bh = region['bbox']
    roi = image[y:y+bh, x:x+bw]
    if roi.size == 0:
        return 'other_damage', 0.3

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Edge features
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0

    # Line detection for scratch identification
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    has_lines = lines is not None and len(lines) > 0
    line_count = len(lines) if has_lines else 0
    line_lengths = [np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines] if has_lines else []
    avg_line_length = np.mean(line_lengths) if line_lengths else 0

    # Color analysis for paint damage
    color_std = np.std(hsv_roi[:, :, 1])  # Saturation std
    value_std = np.std(hsv_roi[:, :, 2])  # Value std

    # Texture analysis for crack detection
    glcm_contrast = np.std(gray_roi)
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

    # Aspect ratio
    aspect_ratio = bw / bh if bh > 0 else 1

    scores = {}
    # Dent: low edge density, medium area, smooth edges
    dent_score = 0.4 * (1 - min(edge_density * 5, 1)) + 0.3 * (1 - min(laplacian_var / 500, 1)) + 0.3 * (0.5 if 0.5 < aspect_ratio < 2 else 0)
    scores['dent'] = dent_score

    # Scratch: high line count, thin, elongated
    scratch_score = 0.5 * min(line_count / 10, 1) + 0.3 * (1 - min(bw * bh / 10000, 1)) + 0.2 * min(avg_line_length / 100, 1)
    scores['scratch'] = scratch_score

    # Crack: high texture variance, irregular edges
    crack_score = 0.4 * min(laplacian_var / 800, 1) + 0.3 * min(glcm_contrast / 100, 1) + 0.3 * (1 if aspect_ratio > 3 or aspect_ratio < 0.33 else 0)
    scores['crack'] = crack_score

    # Broken part: large area, high edge density, high intensity difference
    broken_score = 0.4 * min(region['area'] / (image.shape[0] * image.shape[1] * 0.3), 1) + 0.3 * min(edge_density * 3, 1) + 0.3 * min(region['intensity_diff'] / 60, 1)
    scores['broken_part'] = broken_score

    # Paint damage: high color variance, low edge density
    paint_score = 0.4 * min(color_std / 80, 1) + 0.3 * min(value_std / 80, 1) + 0.3 * (1 - min(edge_density * 3, 1))
    scores['paint_damage'] = paint_score

    # Other: default
    scores['other_damage'] = 0.2

    best_type = max(scores, key=scores.get)
    best_confidence = scores[best_type]

    return best_type, best_confidence


def estimate_damage_location(bbox, image_shape):
    """
    Estimate damage location based on bounding box position on the vehicle.
    Returns (location, confidence).
    """
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / w  # Center x (relative)
    cy = (y + bh / 2) / h  # Center y (relative)

    scores = {}
    # Front bumper: bottom center
    scores['front_bumper'] = 1.0 - (abs(cx - 0.5) * 2) - (1 - cy) if cy > 0.7 else 0
    # Rear bumper: top area
    scores['rear_bumper'] = 1.0 - (abs(cx - 0.5) * 2) - cy if cy < 0.3 else 0
    # Hood: top-center area
    scores['hood'] = 1.0 - (abs(cx - 0.5) * 2) - abs(cy - 0.35) * 2 if 0.2 < cy < 0.5 else 0
    # Windshield: center-top
    scores['windshield'] = 1.0 - abs(cy - 0.25) * 3 if cy < 0.4 and 0.15 < cx < 0.85 else 0
    # Left door: left center
    scores['left_door'] = 1.0 - cx * 2 - abs(cy - 0.5) * 1.5 if cx < 0.4 and 0.2 < cy < 0.8 else 0
    # Right door: right center
    scores['right_door'] = 1.0 - (1 - cx) * 2 - abs(cy - 0.5) * 1.5 if cx > 0.6 and 0.2 < cy < 0.8 else 0
    # Roof: top
    scores['roof'] = 1.0 - cy * 3 - abs(cx - 0.5) if cy < 0.3 else 0
    # Side panel: center
    scores['side_panel'] = 1.0 - abs(cx - 0.5) - abs(cy - 0.5) if 0.3 < cx < 0.7 and 0.2 < cy < 0.8 else 0
    # Other
    scores['other_location'] = 0.15

    best_loc = max(scores, key=scores.get)
    best_confidence = max(scores.values())

    return best_loc, best_confidence


def estimate_severity(region, image):
    """
    Estimate damage severity based on region properties.
    Returns (severity, confidence).
    """
    x, y, bw, bh = region['bbox']
    roi = image[y:y+bh, x:x+bw]
    if roi.size == 0:
        return 'low', 0.3

    h, w = image.shape[:2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Factors
    # 1. Relative area of damage
    relative_area = (bw * bh) / (h * w)

    # 2. Edge density (more edges = more severe)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0

    # 3. Color variance (more color variation = more severe)
    color_std = np.std(hsv_roi[:, :, 1])

    # 4. Texture variance
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

    # Composite severity score (0-1)
    severity_score = (
        0.35 * min(relative_area * 5, 1) +
        0.25 * min(edge_density * 3, 1) +
        0.20 * min(color_std / 60, 1) +
        0.20 * min(laplacian_var / 500, 1)
    )

    if severity_score > 0.7:
        return 'high', min(severity_score * 1.2, 1.0)
    elif severity_score > 0.4:
        return 'medium', severity_score * 1.0
    else:
        return 'low', max(severity_score * 0.8, 0.3)


def analyze_damage_image(image_path):
    """
    Complete analysis pipeline for a single damage image.
    Returns list of annotations and overall confidence.
    """
    img_padded, (orig_w, orig_h, orig_img) = load_image(image_path)
    if img_padded is None:
        return [], 0.0

    padding_top = (TARGET_SIZE - orig_h * min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)) // 2
    padding_left = (TARGET_SIZE - orig_w * min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)) // 2
    scale = min(TARGET_SIZE / orig_w, TARGET_SIZE / orig_h)

    detected_regions = detect_damage_contours(img_padded)

    annotations = []
    overall_conf = 0.0
    valid_count = 0

    for region in detected_regions:
        x, y, bw, bh = region['bbox']
        damage_type, type_conf = estimate_damage_type(region, img_padded)
        location, loc_conf = estimate_damage_location((x, y, bw, bh), img_padded.shape)
        severity, sev_conf = estimate_severity(region, img_padded)

        # Overall confidence for this detection
        detection_conf = (region['confidence'] * 0.4 + type_conf * 0.2 + loc_conf * 0.2 + sev_conf * 0.2)

        # Convert box coordinates from padded to original image coordinates
        # Remove padding and scale back
        x_orig = (x - padding_left) / scale
        y_orig = (y - padding_top) / scale
        bw_orig = bw / scale
        bh_orig = bh / scale

        # Clip to original image bounds
        x_orig = max(0, x_orig)
        y_orig = max(0, y_orig)
        bw_orig = min(orig_w - x_orig, bw_orig)
        bh_orig = min(orig_h - y_orig, bh_orig)

        # Only keep if bounding box has reasonable size
        if bw_orig < 10 or bh_orig < 10:
            continue

        # Convert to YOLO format (normalized center x, center y, width, height)
        x_center = (x_orig + bw_orig / 2) / orig_w
        y_center = (y_orig + bh_orig / 2) / orig_h
        width_norm = bw_orig / orig_w
        height_norm = bh_orig / orig_h

        # Get class IDs
        damage_class_id = CLASS_NAMES.index(damage_type)
        location_class_id = NUM_CLASSES - len(LOCATION_NAMES) - len(SEVERITY_NAMES) + LOCATION_NAMES.index(location)
        severity_class_id = NUM_CLASSES - len(SEVERITY_NAMES) + SEVERITY_NAMES.index(severity)

        annotations.append({
            'damage_type': damage_type,
            'damage_type_id': damage_class_id,
            'damage_location': location,
            'location_id': location_class_id,
            'severity': severity,
            'severity_id': severity_class_id,
            'bbox_orig': (int(x_orig), int(y_orig), int(bw_orig), int(bh_orig)),
            'bbox_yolo': (x_center, y_center, width_norm, height_norm),
            'confidence': detection_conf,
            'type_confidence': type_conf,
            'loc_confidence': loc_conf,
            'sev_confidence': sev_conf,
        })
        overall_conf += detection_conf
        valid_count += 1

    if valid_count > 0:
        overall_conf /= valid_count

    return annotations, overall_conf


def analyze_whole_image(image_path):
    """
    For 'whole' images, verify no damage is present.
    Returns empty annotations if truly whole, or flags if potential damage detected.
    """
    img_padded, (orig_w, orig_h, orig_img) = load_image(image_path)
    if img_padded is None:
        return [], 0.0

    detected_regions = detect_damage_contours(img_padded)

    # Even for whole images, we check if any damage-like regions exist
    annotations = []
    overall_conf = 1.0  # High confidence for no damage

    if detected_regions:
        # Check the top regions - if confidence is very low, ignore
        high_conf_regions = [r for r in detected_regions if r['confidence'] >= 0.5]
        if high_conf_regions:
            # These might be false positives, flag for review
            overall_conf = 0.3  # Low confidence for "no damage" prediction

    return annotations, overall_conf


def generate_yolo_label(annotations, label_path):
    """Write YOLO format label file."""
    if not annotations:
        # Empty file = no objects
        with open(label_path, 'w') as f:
            f.write('')
        return

    lines = []
    for ann in annotations:
        xc, yc, w, h = ann['bbox_yolo']
        # Write damage type detection
        lines.append(f"{ann['damage_type_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        # Write location detection (using same bbox)
        lines.append(f"{ann['location_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        # Write severity detection (using same bbox)
        lines.append(f"{ann['severity_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    with open(label_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def process_dataset():
    """Main processing pipeline."""
    print("=" * 70)
    print("VEHICLE DAMAGE DETECTION - DATASET LABELING PIPELINE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Statistics collectors
    stats = {
        'total_images': 0,
        'labeled_images': 0,
        'flagged_images': 0,
        'damage_types': Counter(),
        'damage_locations': Counter(),
        'severities': Counter(),
        'images_with_annotations': 0,
        'total_annotations': 0,
        'manual_review': [],
    }

    annotation_records = []  # For CSV export

    # Process splits
    splits = {
        'train': {'damage': TRAIN_DAMAGE, 'whole': TRAIN_WHOLE, 'out_img': OUT_IMAGES_TRAIN, 'out_lbl': OUT_LABELS_TRAIN},
        'val': {'damage': VAL_DAMAGE, 'whole': VAL_WHOLE, 'out_img': OUT_IMAGES_VAL, 'out_lbl': OUT_LABELS_VAL},
    }

    for split_name, split_cfg in splits.items():
        print(f"\n{'─' * 70}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'─' * 70}")

        damage_dir = split_cfg['damage']
        whole_dir = split_cfg['whole']
        out_img_dir = split_cfg['out_img']
        out_lbl_dir = split_cfg['out_lbl']

        # Process damage images
        damage_files = sorted(list(damage_dir.glob('*'))) if damage_dir.exists() else []
        print(f"  Damage images: {len(damage_files)}")

        for i, img_path in enumerate(damage_files):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue

            sys.stdout.write(f"\r  [{split_name}] Damage: {i+1}/{len(damage_files)} ({img_path.name})" + ' ' * 20)
            sys.stdout.flush()

            stats['total_images'] += 1
            annotations, confidence = analyze_damage_image(img_path)

            # Copy image to output
            out_img_path = out_img_dir / img_path.name
            shutil.copy2(str(img_path), str(out_img_path))

            if annotations:
                # Check confidence
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Generate YOLO label
                    label_name = img_path.stem + '.txt'
                    label_path = out_lbl_dir / label_name
                    generate_yolo_label(annotations, label_path)

                    stats['labeled_images'] += 1
                    stats['images_with_annotations'] += 1
                    stats['total_annotations'] += len(annotations)

                    for ann in annotations:
                        stats['damage_types'][ann['damage_type']] += 1
                        stats['damage_locations'][ann['damage_location']] += 1
                        stats['severities'][ann['severity']] += 1

                        annotation_records.append({
                            'image_name': img_path.name,
                            'split': split_name,
                            'damage_type': ann['damage_type'],
                            'damage_location': ann['damage_location'],
                            'severity': ann['severity'],
                            'bounding_box': f"[{ann['bbox_orig'][0]}, {ann['bbox_orig'][1]}, {ann['bbox_orig'][2]}, {ann['bbox_orig'][3]}]",
                            'confidence': round(ann['confidence'], 4),
                            'flag': 'auto_labeled',
                        })
                else:
                    # Low confidence - flag for manual review
                    stats['flagged_images'] += 1
                    stats['manual_review'].append(img_path.name)

                    # Still create a label with low confidence flag
                    label_name = img_path.stem + '.txt'
                    label_path = out_lbl_dir / label_name
                    generate_yolo_label(annotations, label_path)

                    annotation_records.append({
                        'image_name': img_path.name,
                        'split': split_name,
                        'damage_type': 'manual_review_needed',
                        'damage_location': 'manual_review_needed',
                        'severity': 'manual_review_needed',
                        'bounding_box': 'manual_review_needed',
                        'confidence': round(confidence, 4),
                        'flag': 'manual_review',
                    })

                    # Copy to manual review folder
                    review_subdir = MANUAL_REVIEW_DIR / split_name
                    review_subdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(img_path), str(review_subdir / img_path.name))
            else:
                # No damage detected in supposed damage image - flag
                stats['flagged_images'] += 1
                stats['manual_review'].append(img_path.name)

                # Create empty label
                label_name = img_path.stem + '.txt'
                label_path = out_lbl_dir / label_name
                with open(label_path, 'w') as f:
                    f.write('')

                annotation_records.append({
                    'image_name': img_path.name,
                    'split': split_name,
                    'damage_type': 'no_damage_detected',
                    'damage_location': 'unknown',
                    'severity': 'unknown',
                    'bounding_box': 'none',
                    'confidence': 0.0,
                    'flag': 'no_detection',
                })

                review_subdir = MANUAL_REVIEW_DIR / split_name
                review_subdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(review_subdir / img_path.name))

        # Process whole images (these should have NO damage annotations)
        whole_files = sorted(list(whole_dir.glob('*'))) if whole_dir.exists() else []
        print(f"\n  Whole images: {len(whole_files)}")

        for i, img_path in enumerate(whole_files):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue

            sys.stdout.write(f"\r  [{split_name}] Whole: {i+1}/{len(whole_files)} ({img_path.name})" + ' ' * 20)
            sys.stdout.flush()

            stats['total_images'] += 1
            annotations, confidence = analyze_whole_image(img_path)

            # Copy image
            out_img_path = out_img_dir / img_path.name
            shutil.copy2(str(img_path), str(out_img_path))

            # Create empty label (no damage = no objects)
            label_name = img_path.stem + '.txt'
            label_path = out_lbl_dir / label_name
            with open(label_path, 'w') as f:
                f.write('')

            if confidence < 0.5:
                stats['flagged_images'] += 1
                stats['manual_review'].append(f"{img_path.name} (whole - potential damage)")
                annotation_records.append({
                    'image_name': img_path.name,
                    'split': split_name,
                    'damage_type': 'whole_potential_damage',
                    'damage_location': 'unknown',
                    'severity': 'unknown',
                    'bounding_box': 'none',
                    'confidence': round(confidence, 4),
                    'flag': 'review_whole',
                })
                review_subdir = MANUAL_REVIEW_DIR / split_name
                review_subdir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(img_path), str(review_subdir / f"REVIEW_{img_path.name}"))
            else:
                annotation_records.append({
                    'image_name': img_path.name,
                    'split': split_name,
                    'damage_type': 'none',
                    'damage_location': 'none',
                    'severity': 'none',
                    'bounding_box': 'none',
                    'confidence': round(confidence, 4),
                    'flag': 'whole_confirmed',
                })

        print()

    # Copy some validation images to test split (20% of validation)
    print("\nCreating test split from validation data...")
    val_images = sorted(list(OUT_IMAGES_VAL.glob('*')))
    val_labels = sorted(list(OUT_LABELS_VAL.glob('*')))
    test_count = max(1, len(val_images) // 5)

    for img_path in val_images[:test_count]:
        shutil.copy2(str(img_path), str(OUT_IMAGES_TEST / img_path.name))
        label_name = img_path.stem + '.txt'
        label_path = OUT_LABELS_VAL / label_name
        if label_path.exists():
            shutil.copy2(str(label_path), str(OUT_LABELS_TEST / label_name))

    print(f"  Copied {test_count} images to test split")

    return stats, annotation_records


def generate_classes_file():
    """Generate classes.txt with all class names."""
    classes_path = OUTPUT_DIR / 'classes.txt'
    with open(classes_path, 'w') as f:
        for i, name in enumerate(ALL_CLASSES):
            f.write(f"{name}\n")
    print(f"[✓] Created classes.txt with {len(ALL_CLASSES)} classes")


def generate_data_yaml():
    """Generate data.yaml compatible with YOLOv8."""
    data_yaml = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': NUM_CLASSES,
        'names': ALL_CLASSES,
        # Damage type sub-classes (0-5)
        'damage_types': {
            'dent': 0,
            'scratch': 1,
            'crack': 2,
            'broken_part': 3,
            'paint_damage': 4,
            'other_damage': 5,
        },
        # Location sub-classes (6-14)
        'damage_locations': {
            'front_bumper': 6,
            'rear_bumper': 7,
            'hood': 8,
            'windshield': 9,
            'left_door': 10,
            'right_door': 11,
            'roof': 12,
            'side_panel': 13,
            'other_location': 14,
        },
        # Severity sub-classes (15-17)
        'severity_levels': {
            'low': 15,
            'medium': 16,
            'high': 17,
        },
    }

    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[✓] Created data.yaml for YOLOv8")


def generate_annotation_csv(records):
    """Export annotations to CSV."""
    csv_path = OUTPUT_DIR / 'annotation_report.csv'
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Created annotation_report.csv with {len(records)} records")
    return df


def generate_class_distribution_report(stats):
    """Generate class distribution report."""
    report_path = OUTPUT_DIR / 'reports' / 'class_distribution.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CLASS DISTRIBUTION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Damage Type Distribution:\n")
        f.write("-" * 40 + "\n")
        for dtype, count in stats['damage_types'].most_common():
            f.write(f"  {dtype:20s}: {count:5d} ({count/max(stats['total_annotations'],1)*100:.1f}%)\n")

        f.write(f"\nTotal annotations: {stats['total_annotations']}\n\n")

        f.write("Damage Location Distribution:\n")
        f.write("-" * 40 + "\n")
        for loc, count in stats['damage_locations'].most_common():
            f.write(f"  {loc:20s}: {count:5d} ({count/max(stats['total_annotations'],1)*100:.1f}%)\n")

        f.write("\nSeverity Distribution:\n")
        f.write("-" * 40 + "\n")
        for sev, count in stats['severities'].most_common():
            f.write(f"  {sev:10s}: {count:5d} ({count/max(stats['total_annotations'],1)*100:.1f}%)\n")

    print(f"[✓] Created class distribution report")


def generate_missing_annotation_report(stats):
    """Generate report of images needing manual review."""
    report_path = OUTPUT_DIR / 'reports' / 'missing_annotations.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MISSING / FLAGGED ANNOTATIONS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total flagged for manual review: {len(stats['manual_review'])}\n\n")
        f.write("Flagged images:\n")
        f.write("-" * 40 + "\n")
        for img_name in stats['manual_review']:
            f.write(f"  • {img_name}\n")
        f.write("\n\n")
        f.write("These images have been copied to:\n")
        f.write(f"  {MANUAL_REVIEW_DIR}/\n")
        f.write("\n")
        f.write("Please review these manually and update the labels accordingly.\n")

    print(f"[✓] Created missing annotation report ({len(stats['manual_review'])} flagged)")


def generate_label_statistics(stats, annotation_records):
    """Generate comprehensive label statistics."""
    df = pd.DataFrame(annotation_records)

    report_path = OUTPUT_DIR / 'reports' / 'label_statistics.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LABEL STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("Dataset Overview:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total images processed: {stats['total_images']}\n")
        f.write(f"  Images with annotations: {stats['images_with_annotations']}\n")
        f.write(f"  Total annotations generated: {stats['total_annotations']}\n")
        f.write(f"  Auto-labeled: {stats['labeled_images']}\n")
        f.write(f"  Flagged for review: {stats['flagged_images']}\n\n")

        # Per-split statistics
        if not df.empty and 'split' in df.columns:
            for split in df['split'].unique():
                split_df = df[df['split'] == split]
                auto = split_df[split_df['flag'] == 'auto_labeled']
                review = split_df[split_df['flag'].str.contains('manual_review|no_detection|review', na=False)]
                f.write(f"  {split.upper()} Split:\n")
                f.write(f"      Total: {len(split_df)}\n")
                f.write(f"      Auto-labeled: {len(auto)}\n")
                f.write(f"      Flagged: {len(review)}\n\n")

        f.write("\nBounding Box Statistics:\n")
        f.write("-" * 40 + "\n")
        if df['flag'].isin(['auto_labeled']).any():
            auto_df = df[df['flag'] == 'auto_labeled']
            confidences = auto_df['confidence'].dropna()
            if len(confidences) > 0:
                f.write(f"  Mean confidence: {confidences.mean():.4f}\n")
                f.write(f"  Median confidence: {confidences.median():.4f}\n")
                f.write(f"  Min confidence: {confidences.min():.4f}\n")
                f.write(f"  Max confidence: {confidences.max():.4f}\n")
                f.write(f"  Conf >= 0.7: {len(confidences[confidences >= 0.7])}/{len(confidences)}\n")
                f.write(f"  Conf >= 0.8: {len(confidences[confidences >= 0.8])}/{len(confidences)}\n")
                f.write(f"  Conf >= 0.9: {len(confidences[confidences >= 0.9])}/{len(confidences)}\n")

    print(f"[✓] Created label statistics report")


def generate_final_report(stats):
    """Generate final summary report."""
    report_path = OUTPUT_DIR / 'final_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("VEHICLE DAMAGE DETECTION - FINAL DATASET REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Total Images:          {stats['total_images']}\n")
        f.write(f"   Total Labeled Images:  {stats['labeled_images']} (auto-generated)\n")
        f.write(f"   Total Annotations:     {stats['total_annotations']}\n")
        f.write(f"   Images Flagged:        {stats['flagged_images']} (needs manual review)\n\n")

        f.write("2. DAMAGE TYPE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for dtype, count in stats['damage_types'].most_common():
            pct = count / max(stats['total_annotations'], 1) * 100
            bar = '#' * int(pct / 2) + '-' * (50 - int(pct / 2))
            f.write(f"   {dtype:20s}: {count:5d} ({pct:5.1f}%) {bar}\n")
        f.write(f"\n   Total: {stats['total_annotations']}\n\n")

        f.write("3. DAMAGE LOCATION DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for loc, count in stats['damage_locations'].most_common():
            pct = count / max(stats['total_annotations'], 1) * 100
            bar = '#' * int(pct / 2) + '-' * (50 - int(pct / 2))
            f.write(f"   {loc:20s}: {count:5d} ({pct:5.1f}%) {bar}\n")
        f.write(f"\n   Total: {stats['total_annotations']}\n\n")

        f.write("4. SEVERITY DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for sev, count in stats['severities'].most_common():
            pct = count / max(stats['total_annotations'], 1) * 100
            bar = '#' * int(pct / 2) + '-' * (50 - int(pct / 2))
            f.write(f"   {sev:10s}: {count:5d} ({pct:5.1f}%) {bar}\n")
        f.write(f"\n   Total: {stats['total_annotations']}\n\n")

        f.write("5. OUTPUT STRUCTURE\n")
        f.write("-" * 40 + "\n")
        f.write(f"   {OUTPUT_DIR}/\n")
        f.write(f"   ├── images/\n")
        f.write(f"   │   ├── train/    ({len(list(OUT_IMAGES_TRAIN.glob('*'))) if OUT_IMAGES_TRAIN.exists() else 0} images)\n")
        f.write(f"   │   ├── val/      ({len(list(OUT_IMAGES_VAL.glob('*'))) if OUT_IMAGES_VAL.exists() else 0} images)\n")
        f.write(f"   │   └── test/     ({len(list(OUT_IMAGES_TEST.glob('*'))) if OUT_IMAGES_TEST.exists() else 0} images)\n")
        f.write(f"   ├── labels/\n")
        f.write(f"   │   ├── train/    ({len(list(OUT_LABELS_TRAIN.glob('*.txt'))) if OUT_LABELS_TRAIN.exists() else 0} labels)\n")
        f.write(f"   │   ├── val/      ({len(list(OUT_LABELS_VAL.glob('*.txt'))) if OUT_LABELS_VAL.exists() else 0} labels)\n")
        f.write(f"   │   └── test/     ({len(list(OUT_LABELS_TEST.glob('*.txt'))) if OUT_LABELS_TEST.exists() else 0} labels)\n")
        f.write(f"   ├── classes.txt         ({len(ALL_CLASSES)} classes)\n")
        f.write(f"   ├── data.yaml           (YOLOv8 config)\n")
        f.write(f"   ├── annotation_report.csv\n")
        f.write(f"   └── reports/\n")
        f.write(f"       ├── class_distribution.txt\n")
        f.write(f"       ├── label_statistics.txt\n")
        f.write(f"       └── missing_annotations.txt\n\n")

        f.write("6. CLASS MAPPING\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Damage Types (0-5):     {', '.join(CLASS_NAMES)}\n")
        f.write(f"   Locations (6-14):       {', '.join(LOCATION_NAMES)}\n")
        f.write(f"   Severity (15-17):       {', '.join(SEVERITY_NAMES)}\n\n")

        f.write("7. NOTES\n")
        f.write("-" * 40 + "\n")
        f.write(f"   • Auto-labeling confidence threshold: {CONFIDENCE_THRESHOLD}\n")
        f.write(f"   • Images below threshold copied to: manual_review/\n")
        f.write(f"   • All labels in YOLO format: class_id x_center y_center width height\n")
        f.write(f"   • Each annotation generates 3 labels (type + location + severity)\n\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[✓] Created final report")


def visualize_samples(n_samples=5):
    """Generate visualization of sample annotations for verification."""
    viz_dir = OUTPUT_DIR / 'reports' / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    damage_files = list(OUT_IMAGES_TRAIN.glob('*'))[:n_samples * 3]
    label_files = [OUT_LABELS_TRAIN / (f.stem + '.txt') for f in damage_files]

    samples_drawn = 0
    for img_path, label_path in zip(damage_files, label_files):
        if samples_drawn >= n_samples:
            break
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Read label
        with open(label_path) as f:
            lines = f.readlines()

        if not lines:
            continue

        colors = [
            (0, 255, 0),    # dent - green
            (0, 0, 255),    # scratch - red
            (255, 0, 0),    # crack - blue
            (0, 255, 255),  # broken - yellow
            (255, 0, 255),  # paint - magenta
            (255, 255, 0),  # other - cyan
            (128, 0, 128),  # location colors
            (0, 128, 128),
            (128, 128, 0),
            (0, 0, 128),
            (128, 0, 0),
            (0, 128, 0),
            (128, 128, 128),
            (64, 64, 64),
            (200, 200, 200),
        ]

        damage_colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0),
                         3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0)}

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            color = damage_colors.get(cls_id, (200, 200, 200)) if cls_id < 6 else (128, 128, 128)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            if cls_id < len(ALL_CLASSES):
                label_text = ALL_CLASSES[cls_id]
                cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)

        viz_path = viz_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(viz_path), img)
        samples_drawn += 1
        print(f"  Visualization saved: {viz_path.name}")

    print(f"[✓] Created {samples_drawn} sample visualizations in {viz_dir}")


def copy_preprocessed_images():
    """Also copy any preprocessed images if they exist."""
    preproc_dir = BASE_DIR / 'datasets' / 'preprocessed_dataset'
    if preproc_dir.exists():
        print("\nFound preprocessed dataset - copying to output...")
        for split in ['training', 'validation']:
            src_damage = preproc_dir / split / '00-damage'
            src_whole = preproc_dir / split / '01-whole'
            if src_damage.exists():
                for f in src_damage.glob('*'):
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(str(f), str(OUT_IMAGES_TRAIN / f.name))
            if src_whole.exists():
                for f in src_whole.glob('*'):
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        shutil.copy2(str(f), str(OUT_IMAGES_TRAIN / f.name))


def main():
    """Main entry point."""
    print("\nStarting Dataset Labeling Pipeline...\n")
    start_time = datetime.now()

    # Setup
    setup_directories()

    # Process all images
    stats, annotation_records = process_dataset()

    # Generate output files
    print("\n" + "=" * 70)
    print("GENERATING OUTPUT FILES")
    print("=" * 70)

    generate_classes_file()
    generate_data_yaml()
    df = generate_annotation_csv(annotation_records)
    generate_class_distribution_report(stats)
    generate_label_statistics(stats, annotation_records)
    generate_missing_annotation_report(stats)
    generate_final_report(stats)

    # Visualize some samples
    print("\nGenerating sample visualizations...")
    visualize_samples(n_samples=5)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Auto-labeled: {stats['labeled_images']}")
    print(f"Flagged for review: {stats['flagged_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"\nNext steps:")
    print(f"  1. Review images in {MANUAL_REVIEW_DIR}/")
    print(f"  2. Update labels for flagged images")
    print(f"  3. Train YOLOv8 with: yolo train data={OUTPUT_DIR}/data.yaml model=models/yolov8n.pt epochs=100")
    print("=" * 70)


if __name__ == '__main__':
    main()
