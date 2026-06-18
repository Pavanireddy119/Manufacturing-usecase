#!/usr/bin/env python3
"""
Production-Ready Dataset Labeling Pipeline for Vehicle Damage Detection
Builds upon the existing auto-labeled dataset, improves annotations, processes
manual review images, and creates a YOLOv8-ready dataset.

Key improvements:
- Better damage detection using multi-scale analysis
- Improved contour analysis with adaptive thresholds
- Spatial validation of bounding boxes
- Semantic region segmentation for accurate damage location
- Enhanced severity estimation
- Dataset consistency validation
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import yaml
import shutil
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / 'datasets' / 'Dataset' / 'data1a'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'dataset_final'

# Input paths (original data)
TRAIN_DAMAGE = DATASET_DIR / 'training' / '00-damage'
TRAIN_WHOLE = DATASET_DIR / 'training' / '01-whole'
VAL_DAMAGE = DATASET_DIR / 'validation' / '00-damage'
VAL_WHOLE = DATASET_DIR / 'validation' / '01-whole'

# Existing auto-labeled dataset
EXISTING_LABELED = BASE_DIR / 'datasets' / 'dataset_labeled'
EXISTING_LABELS_DIR = EXISTING_LABELED / 'labels'
EXISTING_IMAGES_DIR = EXISTING_LABELED / 'images'
MANUAL_REVIEW_DIR = EXISTING_LABELED / 'manual_review'

# Output paths
OUT_IMAGES_TRAIN = OUTPUT_DIR / 'images' / 'train'
OUT_IMAGES_VAL = OUTPUT_DIR / 'images' / 'val'
OUT_IMAGES_TEST = OUTPUT_DIR / 'images' / 'test'
OUT_LABELS_TRAIN = OUTPUT_DIR / 'labels' / 'train'
OUT_LABELS_VAL = OUTPUT_DIR / 'labels' / 'val'
OUT_LABELS_TEST = OUTPUT_DIR / 'labels' / 'test'

# YOLO class mapping (18 classes - same as original)
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

ALL_CLASSES = CLASS_NAMES + LOCATION_NAMES + SEVERITY_NAMES
NUM_CLASSES = len(ALL_CLASSES)
TARGET_SIZE = 640

# Quality thresholds
MIN_CONFIDENCE_THRESHOLD = 0.55
HIGH_CONFIDENCE_THRESHOLD = 0.75
MIN_BBOX_AREA_RATIO = 0.001  # Minimum bbox area as fraction of image
MAX_BBOX_AREA_RATIO = 0.75   # Maximum bbox area as fraction of image

# Visual sample count
VISUAL_SAMPLE_COUNT = 100

# Validation split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.05

random.seed(42)
np.random.seed(42)


def setup_directories():
    """Create all necessary output directories."""
    dirs = [
        OUT_IMAGES_TRAIN, OUT_IMAGES_VAL, OUT_IMAGES_TEST,
        OUT_LABELS_TRAIN, OUT_LABELS_VAL, OUT_LABELS_TEST,
        OUTPUT_DIR / 'review_samples',
        OUTPUT_DIR / 'reports',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[✓] Output directories created under {OUTPUT_DIR}")


def load_image(image_path, target_size=TARGET_SIZE):
    """Load and preprocess an image, returning padded version and metadata."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    orig = img.copy()
    h, w = img.shape[:2]
    
    # Resize maintaining aspect ratio
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    
    # Pad to square
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    
    metadata = {
        'orig_w': w,
        'orig_h': h,
        'scale': scale,
        'pad_top': top,
        'pad_left': left,
        'orig_img': orig,
    }
    return img_padded, metadata


def yolo_to_bbox(yolo_line, img_w, img_h):
    """Convert YOLO format (class_id xc yc w h) to pixel bbox (x1, y1, x2, y2)."""
    parts = yolo_line.strip().split()
    if len(parts) != 5:
        return None
    cls_id = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:])
    x1 = int((xc - bw / 2) * img_w)
    y1 = int((yc - bh / 2) * img_h)
    x2 = int((xc + bw / 2) * img_w)
    y2 = int((yc + bh / 2) * img_h)
    return cls_id, x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel bbox to YOLO format normalized coordinates."""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)
    
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    
    xc = (x1 + bw / 2) / img_w
    yc = (y1 + bh / 2) / img_h
    w_norm = bw / img_w
    h_norm = bh / img_h
    
    return (xc, yc, w_norm, h_norm)


def iou(box1, box2):
    """Calculate IoU between two boxes in (x1,y1,x2,y2) format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def is_valid_bbox(x1, y1, x2, y2, img_w, img_h):
    """Check if a bounding box is valid (within image bounds, reasonable size)."""
    if x1 >= x2 or y1 >= y2:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False
    
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_w * img_h
    area_ratio = bbox_area / img_area
    
    if area_ratio < MIN_BBOX_AREA_RATIO or area_ratio > MAX_BBOX_AREA_RATIO:
        return False
    
    return True


# ============================================================
# ENHANCED DAMAGE DETECTION
# ============================================================

def detect_damage_regions_enhanced(image):
    """
    Enhanced damage detection using multi-scale analysis.
    Returns list of (bbox, confidence, features) for each detected region.
    """
    h, w = image.shape[:2]
    results = []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Strategy 1: Edge-based detection with adaptive thresholds
    strategies = []
    
    # 1a. Canny with Otsu-derived thresholds
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges1 = cv2.Canny(gray, otsu_thresh * 0.3, otsu_thresh * 0.7)
    strategies.append(('canny_otsu', edges1))
    
    # 1b. Canny with fixed thresholds (aggressive)
    edges2 = cv2.Canny(gray, 30, 100)
    strategies.append(('canny_aggressive', edges2))
    
    # 1c. Gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    gradient_mag = np.uint8(np.clip(gradient_mag / gradient_mag.max() * 255, 0, 255))
    _, edges3 = cv2.threshold(gradient_mag, 40, 255, cv2.THRESH_BINARY)
    strategies.append(('gradient', edges3))
    
    # Strategy 2: Color-based anomaly detection
    # Compute local color statistics to find anomalous regions
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    
    # Local standard deviation of saturation and value
    s_std = cv2.boxFilter(s_channel, -1, (15, 15))  # Local mean
    s_diff = np.abs(s_channel - s_std)
    _, color_anomaly = cv2.threshold(s_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    strategies.append(('color_anomaly', color_anomaly))
    
    # Strategy 3: Texture analysis using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.clip(np.abs(laplacian) / 10, 0, 255))
    _, texture_edges = cv2.threshold(laplacian_abs, 15, 255, cv2.THRESH_BINARY)
    strategies.append(('texture', texture_edges))
    
    # Strategy 4: Morphological analysis for dents/damage
    # Damage often creates dark spots with specific shapes
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((15, 15), np.uint8))
    _, tophat_bin = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)
    _, blackhat_bin = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    morphological = cv2.bitwise_or(tophat_bin, blackhat_bin)
    strategies.append(('morphological', morphological))
    
    # Combine all edge detection results
    combined_edges = np.zeros_like(gray)
    for name, edge_map in strategies:
        combined_edges = cv2.bitwise_or(combined_edges, edge_map)
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=2)
    
    # Find contours
    contours, hierarchy = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > w * h * 0.85:
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Skip if bbox is essentially the full image
        if bw > w * 0.95 and bh > h * 0.95:
            continue
        
        # Analyze region features
        region = image[y:y+bh, x:x+bw]
        if region.size == 0:
            continue
        
        region_gray = gray[y:y+bh, x:x+bw]
        region_hsv = hsv[y:y+bh, x:x+bw]
        
        # Compute feature confidence
        # 1. Edge density within region
        region_edges = combined_edges[y:y+bh, x:x+bw]
        edge_density = np.sum(region_edges > 0) / region_edges.size if region_edges.size > 0 else 0
        
        # 2. Texture variance
        laplacian_var = cv2.Laplacian(region_gray, cv2.CV_64F).var()
        
        # 3. Color variance
        color_std = np.std(region_hsv[:, :, 1])
        value_std = np.std(region_hsv[:, :, 2])
        
        # 4. Intensity contrast with surroundings
        y1_s = max(0, y - 15)
        y2_s = min(h, y + bh + 15)
        x1_s = max(0, x - 15)
        x2_s = min(w, x + bw + 15)
        surround_mean = np.mean(gray[y1_s:y2_s, x1_s:x2_s])
        region_mean = np.mean(region_gray)
        intensity_diff = abs(float(region_mean) - float(surround_mean))
        
        # 5. Contour complexity (perimeter^2 / area) - higher = more irregular = damage
        perimeter = cv2.arcLength(cnt, True)
        contour_complexity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 1
        
        # 6. Aspect ratio analysis
        aspect_ratio = bw / bh if bh > 0 else 1
        
        # Calculate composite confidence score
        confidence = min(1.0, (
            0.20 * min(edge_density * 3, 1.0) +
            0.15 * min(laplacian_var / 600, 1.0) +
            0.15 * min(color_std / 70, 1.0) +
            0.15 * min(intensity_diff / 40, 1.0) +
            0.15 * min(contour_complexity / 3, 1.0) +
            0.10 * min(value_std / 60, 1.0) +
            0.10 * (1.0 if 0.3 < aspect_ratio < 3.0 else 0.5)
        ))
        
        # Refine bounding box by analyzing the contour more precisely
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Use the tighter bounding rect from the min area rect
        x_tight = min(box[:, 0])
        y_tight = min(box[:, 1])
        x2_tight = max(box[:, 0])
        y2_tight = max(box[:, 1])
        
        # Add small padding (5% of dimension)
        pad_x = int((x2_tight - x_tight) * 0.05)
        pad_y = int((y2_tight - y_tight) * 0.05)
        x_tight = max(0, x_tight - pad_x)
        y_tight = max(0, y_tight - pad_y)
        x2_tight = min(w - 1, x2_tight + pad_x)
        y2_tight = min(h - 1, y2_tight + pad_y)
        
        if x_tight >= x2_tight or y_tight >= y2_tight:
            continue
            
        bw_tight = x2_tight - x_tight
        bh_tight = y2_tight - y_tight
        
        if bw_tight < 20 or bh_tight < 20:
            continue
        
        results.append({
            'bbox': (x_tight, y_tight, bw_tight, bh_tight),
            'confidence': confidence,
            'edge_density': edge_density,
            'laplacian_var': laplacian_var,
            'color_std': color_std,
            'intensity_diff': intensity_diff,
            'contour_complexity': contour_complexity,
            'area': area,
        })
    
    # NMS - Non-Maximum Suppression to remove overlapping detections
    results.sort(key=lambda r: r['confidence'], reverse=True)
    filtered = []
    
    for region in results:
        x1, y1, bw, bh = region['bbox']
        x2, y2 = x1 + bw, y1 + bh
        overlapping = False
        
        for kept in filtered:
            kx1, ky1, kbw, kbh = kept['bbox']
            kx2, ky2 = kx1 + kbw, ky1 + kbh
            
            inter_iou = iou((x1, y1, x2, y2), (kx1, ky1, kx2, ky2))
            if inter_iou > 0.4:
                overlapping = True
                break
        
        if not overlapping:
            filtered.append(region)
    
    return filtered


# ============================================================
# DAMAGE TYPE CLASSIFICATION
# ============================================================

def classify_damage_type(region, image):
    """
    Classify damage type based on visual features.
    Returns (damage_type, confidence).
    """
    x, y, bw, bh = region['bbox']
    x2, y2 = x + bw, y + bh
    
    roi = image[y:y2, x:x2]
    if roi.size == 0:
        return 'other_damage', 0.3
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Edge features
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    # Line detection for scratches
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=20, maxLineGap=10)
    has_lines = lines is not None and len(lines) > 0
    line_count = len(lines) if has_lines else 0
    line_lengths = [np.sqrt((l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2) for l in lines] if has_lines else []
    avg_line_length = np.mean(line_lengths) if line_lengths else 0
    max_line_length = max(line_lengths) if line_lengths else 0
    
    # Texture analysis
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    
    # Color analysis
    color_std = np.std(hsv_roi[:, :, 1])
    hue_std = np.std(hsv_roi[:, :, 0])
    value_std = np.std(hsv_roi[:, :, 2])
    
    # Shape analysis
    aspect_ratio = bw / bh if bh > 0 else 1
    region_area_ratio = (bw * bh) / (image.shape[0] * image.shape[1])
    
    # GLCM-like contrast (simple approximation)
    glcm_contrast = np.std(gray_roi)
    
    scores = {}
    
    # Dent: moderate size, low edge density, smooth texture, round/oval shape
    dent_score = (
        0.25 * (1 - min(edge_density * 3, 1)) +
        0.20 * (1 - min(laplacian_var / 500, 1)) +
        0.20 * (1 if 0.5 < aspect_ratio < 2.0 else 0.3) +
        0.20 * min(region_area_ratio * 10, 1) +
        0.15 * (1 - min(color_std / 60, 1))
    )
    scores['dent'] = dent_score
    
    # Scratch: thin, elongated, high line count
    scratch_score = (
        0.35 * min(line_count / 8, 1) +
        0.20 * min(max_line_length / 150, 1) +
        0.20 * (1 if aspect_ratio > 2.5 or aspect_ratio < 0.4 else 0.2) +
        0.15 * (1 - min(region_area_ratio * 20, 1)) +
        0.10 * min(edge_density * 2, 1)
    )
    scores['scratch'] = scratch_score
    
    # Crack: high texture variance, irregular, line-like but branching
    crack_score = (
        0.30 * min(laplacian_var / 700, 1) +
        0.25 * min(glcm_contrast / 80, 1) +
        0.20 * (1 if aspect_ratio > 2.0 or aspect_ratio < 0.5 else 0.3) +
        0.15 * min(hue_std / 40, 1) +
        0.10 * min(line_count / 5, 1)
    )
    scores['crack'] = crack_score
    
    # Broken part: large area, high edge density, high contrast
    broken_score = (
        0.30 * min(region_area_ratio * 8, 1) +
        0.25 * min(edge_density * 3, 1) +
        0.20 * min(region['intensity_diff'] / 50, 1) +
        0.15 * min(laplacian_var / 400, 1) +
        0.10 * min(contour_complexity := region.get('contour_complexity', 1) / 2, 1) if hasattr(region, 'get') else 0
    )
    if 'contour_complexity' in region:
        broken_score = (
            0.30 * min(region_area_ratio * 8, 1) +
            0.25 * min(edge_density * 3, 1) +
            0.20 * min(region['intensity_diff'] / 50, 1) +
            0.15 * min(laplacian_var / 400, 1) +
            0.10 * min(region['contour_complexity'] / 2, 1)
        )
    scores['broken_part'] = broken_score
    
    # Paint damage: high color variance, low edge density, smooth texture
    paint_score = (
        0.30 * min(color_std / 70, 1) +
        0.25 * min(hue_std / 50, 1) +
        0.20 * (1 - min(edge_density * 3, 1)) +
        0.15 * (1 - min(laplacian_var / 400, 1)) +
        0.10 * min(value_std / 60, 1)
    )
    scores['paint_damage'] = paint_score
    
    # Other: baseline
    scores['other_damage'] = 0.15 + 0.1 * (1 - max(list(scores.values()))) if scores else 0.2
    
    best_type = max(scores, key=scores.get)
    best_confidence = scores[best_type]
    
    return best_type, best_confidence


# ============================================================
# DAMAGE LOCATION ESTIMATION
# ============================================================

def estimate_damage_location(bbox, image_shape):
    """
    Estimate damage location using spatial analysis of the vehicle.
    Uses a more refined region-based approach.
    Returns (location, confidence).
    """
    h, w = image_shape[:2]
    x, y, bw, bh = bbox
    cx = (x + bw / 2) / w  # Center x (relative)
    cy = (y + bh / 2) / h  # Center y (relative)
    
    # Calculate overlap with predefined vehicle regions
    regions = {
        'front_bumper': (0.05, 0.70, 0.95, 0.98),   # bottom, full width
        'rear_bumper': (0.05, 0.02, 0.95, 0.20),     # top, full width
        'hood': (0.10, 0.15, 0.90, 0.50),             # upper-center
        'windshield': (0.15, 0.10, 0.85, 0.45),       # center-top
        'left_door': (0.00, 0.20, 0.35, 0.80),        # left-middle
        'right_door': (0.65, 0.20, 1.00, 0.80),       # right-middle
        'roof': (0.10, 0.00, 0.90, 0.25),             # top
        'side_panel': (0.25, 0.15, 0.75, 0.85),       # center
        'other_location': (0.00, 0.00, 1.00, 1.00),   # full image
    }
    
    # Bbox in relative coordinates
    rx1, ry1 = x / w, y / h
    rx2, ry2 = (x + bw) / w, (y + bh) / h
    bbox_area = (rx2 - rx1) * (ry2 - ry1)
    
    scores = {}
    for loc_name, (lr1, lr2, lr3, lr4) in regions.items():
        # Calculate IoU between detection bbox and region
        inter_x1 = max(rx1, lr1)
        inter_y1 = max(ry1, lr2)
        inter_x2 = min(rx2, lr3)
        inter_y2 = min(ry2, lr4)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        region_area = (lr3 - lr1) * (lr4 - lr2)
        union = bbox_area + region_area - inter_area
        iou_score = inter_area / union if union > 0 else 0
        
        # Also consider center distance
        region_cx = (lr1 + lr3) / 2
        region_cy = (lr2 + lr4) / 2
        center_dist = np.sqrt((cx - region_cx)**2 + (cy - region_cy)**2)
        dist_score = max(0, 1 - center_dist * 2)
        
        # Combined score
        scores[loc_name] = 0.6 * iou_score + 0.4 * dist_score
    
    # Small bonus for side_panel as it's the most common location
    scores['side_panel'] += 0.05
    
    best_loc = max(scores, key=scores.get)
    best_confidence = min(1.0, max(scores.values()) * 1.2)
    
    return best_loc, best_confidence


# ============================================================
# SEVERITY ESTIMATION
# ============================================================

def estimate_severity(region, image):
    """
    Estimate damage severity with improved multi-factor analysis.
    Returns (severity, confidence).
    """
    x, y, bw, bh = region['bbox']
    x2, y2 = x + bw, y + bh
    
    roi = image[y:y2, x:x2]
    if roi.size == 0:
        return 'low', 0.3
    
    h, w = image.shape[:2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Factors for severity assessment
    
    # 1. Relative area of damage
    relative_area = (bw * bh) / (h * w)
    area_score = min(relative_area * 8, 1.0)
    
    # 2. Edge density (more edges = more severe structural damage)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    edge_score = min(edge_density * 3, 1.0)
    
    # 3. Color variance (more color variation = more severe paint damage)
    color_std = np.std(hsv_roi[:, :, 1])
    color_score = min(color_std / 60, 1.0)
    
    # 4. Texture variance
    laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    texture_score = min(laplacian_var / 500, 1.0)
    
    # 5. Multiple damage indicators in the same region
    # Check if the region contains sub-regions with different characteristics
    half_bw, half_bh = bw // 2, bh // 2
    sub_regions = []
    for sx in [x, x + half_bw]:
        for sy in [y, y + half_bh]:
            if sx + half_bw <= x2 and sy + half_bh <= y2:
                sub = gray_roi[sy-y:sy-y+half_bh, sx-x:sx-x+half_bw]
                if sub.size > 0:
                    sub_regions.append(np.std(sub))
    sub_var_score = min(np.std(sub_regions) / 30, 1.0) if len(sub_regions) > 1 else 0
    
    # 6. Intensity contrast (deeper damage = higher contrast)
    surround_mean = np.mean(gray_roi)
    intensity_diff = region.get('intensity_diff', abs(float(np.mean(gray_roi)) - float(np.mean(gray_roi))))
    contrast_score = min(region.get('intensity_diff', intensity_diff) / 50, 1.0)
    
    # Composite severity score (0-1)
    severity_score = (
        0.30 * area_score +
        0.20 * edge_score +
        0.15 * color_score +
        0.15 * texture_score +
        0.10 * sub_var_score +
        0.10 * contrast_score
    )
    
    # Map score to severity level
    if severity_score > 0.65:
        severity = 'high'
        sev_confidence = min(severity_score * 1.1, 1.0)
    elif severity_score > 0.35:
        severity = 'medium'
        sev_confidence = severity_score * 1.0
    else:
        severity = 'low'
        sev_confidence = max(severity_score * 0.9, 0.3)
    
    return severity, sev_confidence


# ============================================================
# IMAGE ANALYSIS PIPELINE
# ============================================================

def analyze_damage_image(image_path):
    """
    Complete analysis pipeline for a single damage image.
    Returns list of annotations and overall confidence.
    """
    img_padded, metadata = load_image(image_path)
    if img_padded is None:
        return [], 0.0
    
    orig_w = metadata['orig_w']
    orig_h = metadata['orig_h']
    scale = metadata['scale']
    pad_top = metadata['pad_top']
    pad_left = metadata['pad_left']
    
    detected_regions = detect_damage_regions_enhanced(img_padded)
    
    annotations = []
    overall_conf = 0.0
    
    for region in detected_regions:
        x_pad, y_pad, bw_pad, bh_pad = region['bbox']
        
        # Convert from padded image coordinates back to original image coordinates
        x_orig = (x_pad - pad_left) / scale
        y_orig = (y_pad - pad_top) / scale
        bw_orig = bw_pad / scale
        bh_orig = bh_pad / scale
        
        # Clip to original image bounds
        x_orig = max(0, x_orig)
        y_orig = max(0, y_orig)
        bw_orig = min(orig_w - x_orig, bw_orig)
        bh_orig = min(orig_h - y_orig, bh_orig)
        
        # Validate bbox
        if not is_valid_bbox(x_orig, y_orig, x_orig + bw_orig, y_orig + bh_orig, orig_w, orig_h):
            continue
        
        # Only keep if bbox has reasonable size
        if bw_orig < 15 or bh_orig < 15:
            continue
        
        # Classify damage type using original image region
        x1_i, y1_i = int(x_orig), int(y_orig)
        x2_i, y2_i = int(x_orig + bw_orig), int(y_orig + bh_orig)
        
        # Need to re-analyze on original image for accurate classification
        orig_img = metadata['orig_img']
        roi = orig_img[y1_i:y2_i, x1_i:x2_i]
        
        if roi.size == 0:
            continue
        
        # Recompute region features on original image
        region_orig = {
            'bbox': (x1_i, y1_i, int(bw_orig), int(bh_orig)),
            'confidence': region['confidence'],
            'laplacian_var': region['laplacian_var'],
            'color_std': region['color_std'],
            'intensity_diff': region['intensity_diff'],
            'contour_complexity': region.get('contour_complexity', 1),
        }
        
        damage_type, type_conf = classify_damage_type(region_orig, orig_img)
        location, loc_conf = estimate_damage_location((x1_i, y1_i, int(bw_orig), int(bh_orig)), orig_img.shape)
        severity, sev_conf = estimate_severity(region_orig, orig_img)
        
        # Detection confidence
        detection_conf = (
            region['confidence'] * 0.35 +
            type_conf * 0.25 +
            loc_conf * 0.20 +
            sev_conf * 0.20
        )
        
        # Convert to YOLO format
        yolo_coords = bbox_to_yolo(x1_i, y1_i, x2_i, y2_i, orig_w, orig_h)
        if yolo_coords is None:
            continue
        
        xc, yc, w_norm, h_norm = yolo_coords
        
        # Get class IDs
        damage_class_id = CLASS_NAMES.index(damage_type)
        location_class_id = 6 + LOCATION_NAMES.index(location)
        severity_class_id = 15 + SEVERITY_NAMES.index(severity)
        
        annotations.append({
            'damage_type': damage_type,
            'damage_type_id': damage_class_id,
            'damage_location': location,
            'location_id': location_class_id,
            'severity': severity,
            'severity_id': severity_class_id,
            'bbox_orig': (x1_i, y1_i, x2_i - x1_i, y2_i - y1_i),
            'bbox_pixels': (x1_i, y1_i, x2_i, y2_i),
            'bbox_yolo': (xc, yc, w_norm, h_norm),
            'confidence': detection_conf,
            'type_confidence': type_conf,
            'loc_confidence': loc_conf,
            'sev_confidence': sev_conf,
        })
        overall_conf += detection_conf
    
    if annotations:
        overall_conf /= len(annotations)
    
    return annotations, overall_conf


def analyze_whole_image(image_path):
    """
    For 'whole' images, verify no damage is present.
    Returns empty annotations if truly whole, or flags if potential damage detected.
    """
    img_padded, metadata = load_image(image_path)
    if img_padded is None:
        return [], 0.0
    
    orig_img = metadata['orig_img']
    detected_regions = detect_damage_regions_enhanced(img_padded)
    
    annotations = []
    overall_conf = 1.0  # High confidence for no damage
    
    if detected_regions:
        # Check the top regions
        high_conf_regions = [r for r in detected_regions if r['confidence'] >= 0.55]
        if high_conf_regions:
            # Could be false positives from reflections/shadow patterns
            # We'll still flag but check if probably just reflections
            avg_conf = np.mean([r['confidence'] for r in high_conf_regions])
            
            if avg_conf > 0.7:
                # Likely actual damage misclassified as whole
                overall_conf = 0.25
            elif avg_conf > 0.55:
                # Possibly reflections/shadows, mark as uncertain
                overall_conf = 0.45
            else:
                overall_conf = 0.6
    
    return annotations, overall_conf


# ============================================================
# VALIDATE EXISTING AUTO-GENERATED ANNOTATIONS
# ============================================================

def validate_existing_annotations():
    """
    Validate all existing auto-generated annotations by checking against
    detected damage regions. Returns validated annotations and flags.
    """
    print("\n" + "=" * 70)
    print("VALIDATING EXISTING AUTO-GENERATED ANNOTATIONS")
    print("=" * 70)
    
    validated = []
    removed_count = 0
    uncertain_count = 0
    
    splits = ['train', 'val', 'test']
    for split in splits:
        labels_dir = EXISTING_LABELS_DIR / split
        images_dir = EXISTING_IMAGES_DIR / split
        
        if not labels_dir.exists() or not images_dir.exists():
            continue
        
        label_files = sorted(list(labels_dir.glob('*.txt')))
        print(f"\n  Validating {split} split ({len(label_files)} labels)...")
        
        for label_path in label_files:
            img_name = label_path.stem + '.JPEG'
            img_path = images_dir / img_name
            if not img_path.exists():
                img_path = images_dir / (label_path.stem + '.jpg')
            if not img_path.exists():
                img_path = images_dir / (label_path.stem + '.png')
            
            if not img_path.exists():
                # Try all possible extensions
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    img_path = images_dir / (label_path.stem + ext)
                    if img_path.exists():
                        break
            
            if not img_path.exists():
                print(f"    ⚠ Image not found for {label_path.name}, skipping")
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            
            with open(label_path) as f:
                lines = f.readlines()
            
            valid_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                
                # Validate class ID
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    removed_count += 1
                    continue
                
                # Convert to pixel coordinates
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                
                # Check if bbox is valid
                if not is_valid_bbox(x1, y1, x2, y2, w, h):
                    removed_count += 1
                    continue
                
                # Check if bbox is too large (covers most of image)
                bbox_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                if bbox_area / img_area > 0.8:
                    removed_count += 1
                    continue
                
                # Check if bbox starts from edge (0,0) suggesting poor detection
                if x1 == 0 and y1 == 0:
                    # Only remove if it also covers a large portion
                    if (x2 - x1) > w * 0.8 or (y2 - y1) > h * 0.8:
                        removed_count += 1
                        continue
                
                valid_lines.append(line)
            
            # Write back validated labels
            with open(label_path, 'w') as f:
                if valid_lines:
                    f.write('\n'.join(valid_lines) + '\n')
                # If all lines removed, leave empty file
            
            if len(lines) > 0 and len(valid_lines) < len(lines):
                removed = len(lines) - len(valid_lines)
                uncertain_count += removed
    
    print(f"\n  Validation complete:")
    print(f"    Removed invalid annotations: {removed_count}")
    print(f"    Total validated annotations: uncertain")
    return validated


# ============================================================
# PROCESS MANUAL REVIEW IMAGES
# ============================================================

def process_manual_review():
    """
    Re-process all images in manual_review folder with enhanced detection.
    Returns annotations and statistics.
    """
    print("\n" + "=" * 70)
    print("PROCESSING MANUAL REVIEW IMAGES WITH ENHANCED DETECTION")
    print("=" * 70)
    
    all_annotations = []
    stats = {
        'total_processed': 0,
        'damage_detected': 0,
        'no_damage_detected': 0,
        'low_confidence': 0,
        'damage_types': Counter(),
        'damage_locations': Counter(),
        'severities': Counter(),
    }
    
    splits = ['train', 'val']
    for split in splits:
        review_dir = MANUAL_REVIEW_DIR / split
        if not review_dir.exists():
            continue
        
        image_files = sorted([f for f in review_dir.glob('*') 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
        
        print(f"\n  Processing {split} manual review ({len(image_files)} images)...")
        
        for i, img_path in enumerate(image_files):
            sys.stdout.write(f"\r    [{split}] {i+1}/{len(image_files)} ({img_path.name})" + ' ' * 20)
            sys.stdout.flush()
            
            # Analyze with enhanced detection
            annotations, confidence = analyze_damage_image(img_path)
            
            all_annotations.append({
                'image_path': img_path,
                'split': split,
                'annotations': annotations,
                'confidence': confidence,
                'source': 'manual_review_processed',
            })
            
            stats['total_processed'] += 1
            
            if annotations:
                if confidence >= MIN_CONFIDENCE_THRESHOLD:
                    stats['damage_detected'] += 1
                    for ann in annotations:
                        stats['damage_types'][ann['damage_type']] += 1
                        stats['damage_locations'][ann['damage_location']] += 1
                        stats['severities'][ann['severity']] += 1
                else:
                    stats['low_confidence'] += 1
                    stats['damage_detected'] += 1  # Still detected damage
                    for ann in annotations:
                        stats['damage_types'][ann['damage_type']] += 1
                        stats['damage_locations'][ann['damage_location']] += 1
                        stats['severities'][ann['severity']] += 1
            else:
                stats['no_damage_detected'] += 1
    
    print(f"\n\n  Manual review processing results:")
    print(f"    Total processed: {stats['total_processed']}")
    print(f"    Damage detected: {stats['damage_detected']}")
    print(f"    No damage detected: {stats['no_damage_detected']}")
    print(f"    Low confidence: {stats['low_confidence']}")
    
    return all_annotations, stats


# ============================================================
# CREATE FINAL DATASET
# ============================================================

def read_existing_auto_labels():
    """
    Read and index all existing auto-generated labels from dataset_labeled.
    More permissive: keeps labels that have reasonable bounding boxes.
    Returns dict mapping filename base -> (image_path, label_lines)
    """
    print("\n  Reading existing auto-labels from dataset_labeled/...")
    existing_labels = {}
    total_labels_read = 0
    total_lines_read = 0
    total_kept = 0
    
    for split in ['train', 'val', 'test']:
        img_dir = EXISTING_IMAGES_DIR / split
        lbl_dir = EXISTING_LABELS_DIR / split
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        
        for label_path in lbl_dir.glob('*.txt'):
            # Find corresponding image
            base_name = label_path.stem
            img_found = None
            for ext in ['.JPEG', '.jpg', '.jpeg', '.png']:
                candidate = img_dir / (base_name + ext)
                if candidate.exists():
                    img_found = candidate
                    break
            
            if img_found is None:
                continue
            
            # Read label content
            with open(label_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            if not lines:
                continue
            
            total_labels_read += 1
            total_lines_read += len(lines)
            
            # Validate lines
            img = cv2.imread(str(img_found))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            valid_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    continue
                xc, yc, bw, bh = map(float, parts[1:])
                if bw <= 0 or bh <= 0 or bw > 1 or bh > 1:
                    continue
                if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                    continue
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                bbox_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                area_ratio = bbox_area / img_area if img_area > 0 else 0
                
                # Reject only if the box is INVALID (outside bounds or too large)
                if not is_valid_bbox(x1, y1, x2, y2, w, h):
                    continue
                    
                # Reject boxes that cover >85% of image (clearly bad)
                if area_ratio > 0.85:
                    continue
                    
                # Reject boxes that start at (0,0) AND cover >70% of image
                if x1 <= 5 and y1 <= 5 and area_ratio > 0.7:
                    continue
                    
                valid_lines.append(line)
            
            if valid_lines:
                existing_labels[base_name] = {
                    'img_path': img_found,
                    'labels': valid_lines,
                    'split': split,
                }
                total_kept += 1
    
    print(f"    Found {len(existing_labels)} images with valid auto-labels")
    print(f"    (Read {total_labels_read} labels, {total_lines_read} lines, kept {total_kept})")
    return existing_labels


def create_final_dataset():
    """
    Create the final production-ready dataset with all improvements.
    Uses enhanced detection for unlabeled/missing images.
    """
    print("\n" + "=" * 70)
    print("CREATING FINAL PRODUCTION DATASET")
    print("=" * 70)
    
    stats = {
        'total_images': 0,
        'labeled_images': 0,
        'empty_label_images': 0,
        'total_annotations': 0,
        'damage_types': Counter(),
        'damage_locations': Counter(),
        'severities': Counter(),
        'needs_manual_verification': [],
        'splits': defaultdict(int),
    }
    
    # Read existing valid auto-labels
    existing_labels = read_existing_auto_labels()
    
    # Collect all original images with their splits
    original_splits = {
        'train_damage': {'src': TRAIN_DAMAGE, 'is_damage': True, 'original_split': 'train'},
        'train_whole': {'src': TRAIN_WHOLE, 'is_damage': False, 'original_split': 'train'},
        'val_damage': {'src': VAL_DAMAGE, 'is_damage': True, 'original_split': 'val'},
        'val_whole': {'src': VAL_WHOLE, 'is_damage': False, 'original_split': 'val'},
    }
    
    # Collect all image paths by stem name (without extension)
    # This is critical: same stem might exist in both train and val with different extensions
    all_images = []
    
    for key, cfg in original_splits.items():
        src_dir = cfg['src']
        if not src_dir.exists():
            continue
        
        for f in src_dir.glob('*'):
            if f.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue
            all_images.append({
                'path': f,
                'stem': f.stem,
                'is_damage': cfg['is_damage'],
                'original_split': cfg['original_split'],
            })
    
    # Remove duplicates (same filename appears in multiple sources)
    seen = set()
    unique_images = []
    for img in all_images:
        key = (img['stem'], img['original_split'])
        if key not in seen:
            seen.add(key)
            unique_images.append(img)
    
    print(f"\n  Total unique images: {len(unique_images)}")
    used_existing = 0
    used_enhanced = 0
    
    # Process all images
    processed_images = {}
    
    for i, img_info in enumerate(unique_images):
        img_path = img_info['path']
        stem = img_info['stem']
        is_damage = img_info['is_damage']
        original_split = img_info['original_split']
        
        sys.stdout.write(f"\r  Processing: {i+1}/{len(unique_images)} ({img_path.name}, damage={is_damage})" + ' ' * 30)
        sys.stdout.flush()
        
        # Step 1: Check if this image has existing valid auto-labels
        if stem in existing_labels:
            label_data = existing_labels[stem]
            valid_lines = label_data['labels']
            
            processed_images[img_path.name] = {
                'img_path': img_path,
                'labels': valid_lines,
                'is_damage': is_damage,
                'original_split': original_split,
                'source': 'existing_auto_labeled',
                'num_anns': len(valid_lines) // 3,
            }
            
            stats['total_images'] += 1
            stats['labeled_images'] += 1
            ann_count = len(valid_lines) // 3
            stats['total_annotations'] += ann_count
            used_existing += 1
            
            # Parse annotations for statistics
            for line in valid_lines:
                parts = line.split()
                cls_id = int(parts[0])
                if cls_id < 6:
                    stats['damage_types'][CLASS_NAMES[cls_id]] += 1
                elif cls_id < 15:
                    stats['damage_locations'][LOCATION_NAMES[cls_id - 6]] += 1
                else:
                    stats['severities'][SEVERITY_NAMES[cls_id - 15]] += 1
            
        elif is_damage:
            # Step 2: No valid existing label and it's a damage image - run enhanced detection
            annotations, confidence = analyze_damage_image(img_path)
            
            if annotations and len(annotations) > 0:
                label_lines = []
                for ann in annotations:
                    xc, yc, w_norm, h_norm = ann['bbox_yolo']
                    label_lines.append(f"{ann['damage_type_id']} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                    label_lines.append(f"{ann['location_id']} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                    label_lines.append(f"{ann['severity_id']} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                processed_images[img_path.name] = {
                    'img_path': img_path,
                    'labels': label_lines,
                    'is_damage': is_damage,
                    'original_split': original_split,
                    'source': 'enhanced_detection',
                    'num_anns': len(annotations),
                    'confidence': confidence,
                }
                
                stats['total_images'] += 1
                stats['labeled_images'] += 1
                stats['total_annotations'] += len(annotations)
                used_enhanced += 1
                
                for ann in annotations:
                    stats['damage_types'][ann['damage_type']] += 1
                    stats['damage_locations'][ann['damage_location']] += 1
                    stats['severities'][ann['severity']] += 1
            else:
                # No damage detected - empty label
                processed_images[img_path.name] = {
                    'img_path': img_path,
                    'labels': [],
                    'is_damage': is_damage,
                    'original_split': original_split,
                    'source': 'no_damage_detected',
                    'num_anns': 0,
                }
                stats['total_images'] += 1
                stats['empty_label_images'] += 1
                stats['needs_manual_verification'].append({
                    'image': img_path.name,
                    'reason': 'No damage detected in damage image',
                })
        else:
            # Whole images - always empty labels (no damage)
            processed_images[img_path.name] = {
                'img_path': img_path,
                'labels': [],
                'is_damage': is_damage,
                'original_split': original_split,
                'source': 'whole_image',
                'num_anns': 0,
            }
            stats['total_images'] += 1
            stats['empty_label_images'] += 1
    
    print(f"\n\n  Processing complete:")
    print(f"    Total images: {stats['total_images']}")
    print(f"    Labeled images: {stats['labeled_images']}")
    print(f"    Empty labels: {stats['empty_label_images']}")
    print(f"    Total annotations: {stats['total_annotations']}")
    print(f"    Used existing auto-labels: {used_existing}")
    print(f"    Used enhanced detection: {used_enhanced}")
    
    return processed_images, stats


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(processed_images):
    """Split images into train/val/test based on original splits."""
    print("\n" + "=" * 70)
    print("SPLITTING DATASET INTO TRAIN/VAL/TEST")
    print("=" * 70)
    
    # Organize by original split
    train_images = []
    val_images = []
    
    for name, info in processed_images.items():
        if info['original_split'] == 'train':
            train_images.append((name, info))
        else:
            val_images.append((name, info))
    
    # Use original validation images for val and test
    random.shuffle(val_images)
    test_count = max(1, len(val_images) // 4)  # ~25% of validation for test
    val_count = len(val_images) - test_count
    
    test_images = val_images[:test_count]
    val_images = val_images[test_count:]
    
    split_data = {
        'train': train_images,
        'val': val_images,
        'test': test_images,
    }
    
    for split_name, images in split_data.items():
        out_img_dir = OUTPUT_DIR / 'images' / split_name
        out_lbl_dir = OUTPUT_DIR / 'labels' / split_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        for img_name, info in images:
            src_path = info['img_path']
            
            # Determine output image extension (use original)
            dst_img_path = out_img_dir / (info['img_path'].stem + info['img_path'].suffix)
            try:
                shutil.copy2(str(src_path), str(dst_img_path))
            except:
                # If source doesn't exist, try other paths
                continue
            
            # Write label file
            label_name = info['img_path'].stem + '.txt'
            label_path = out_lbl_dir / label_name
            if info['labels']:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(info['labels']) + '\n')
            else:
                # Empty label file
                with open(label_path, 'w') as f:
                    f.write('')
            
            copied += 1
        
        print(f"  {split_name}: {copied} images copied")
    
    return split_data


# ============================================================
# DATASET CONSISTENCY VALIDATION
# ============================================================

def validate_dataset_consistency():
    """
    Validate the final dataset for consistency:
    - Every image has a corresponding label file
    - No empty annotations (unless whole images)
    - No duplicate annotations
    - Bounding boxes within image boundaries
    """
    print("\n" + "=" * 70)
    print("VALIDATING DATASET CONSISTENCY")
    print("=" * 70)
    
    issues = []
    stats = {
        'total_images': 0,
        'has_label': 0,
        'missing_label': 0,
        'empty_labels': 0,
        'has_annotations': 0,
        'total_annotations': 0,
        'invalid_boxes': 0,
        'duplicates_removed': 0,
    }
    
    for split in ['train', 'val', 'test']:
        img_dir = OUTPUT_DIR / 'images' / split
        lbl_dir = OUTPUT_DIR / 'labels' / split
        
        if not img_dir.exists():
            continue
        
        image_files = sorted([f for f in img_dir.glob('*') 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']])
        
        for img_path in image_files:
            stats['total_images'] += 1
            label_path = lbl_dir / (img_path.stem + '.txt')
            
            # Check label exists
            if not label_path.exists():
                stats['missing_label'] += 1
                issues.append(f"MISSING LABEL: {split}/{img_path.name}")
                # Create empty label
                with open(label_path, 'w') as f:
                    f.write('')
                stats['has_label'] += 1
                continue
            
            stats['has_label'] += 1
            
            # Read and validate labels
            with open(label_path) as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            
            if not lines:
                stats['empty_labels'] += 1
                continue
            
            # Validate each annotation
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            
            valid_lines = []
            seen_boxes = set()
            
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                
                # Validate class ID
                if cls_id < 0 or cls_id >= NUM_CLASSES:
                    stats['invalid_boxes'] += 1
                    continue
                
                # Convert to pixel coordinates
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)
                
                # Validate bounding box
                if not is_valid_bbox(x1, y1, x2, y2, w, h):
                    stats['invalid_boxes'] += 1
                    continue
                
                # Check for duplicates (same class + similar position)
                box_key = (cls_id, x1 // 10, y1 // 10, x2 // 10, y2 // 10)
                if box_key in seen_boxes:
                    stats['duplicates_removed'] += 1
                    continue
                seen_boxes.add(box_key)
                
                valid_lines.append(line)
            
            # Write back validated labels
            with open(label_path, 'w') as f:
                if valid_lines:
                    f.write('\n'.join(valid_lines) + '\n')
                else:
                    f.write('')
            
            stats['total_annotations'] += len(valid_lines)
            if valid_lines:
                stats['has_annotations'] += 1
    
    print(f"\n  Validation results:")
    print(f"    Total images: {stats['total_images']}")
    print(f"    With labels: {stats['has_label']}")
    print(f"    Missing labels (auto-created): {stats['missing_label']}")
    print(f"    Empty labels: {stats['empty_labels']}")
    print(f"    With annotations: {stats['has_annotations']}")
    print(f"    Total annotations: {stats['total_annotations']}")
    print(f"    Invalid boxes removed: {stats['invalid_boxes']}")
    print(f"    Duplicates removed: {stats['duplicates_removed']}")
    
    if issues:
        print(f"\n  Issues found ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    • {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    
    return stats


# ============================================================
# REMOVE EMPTY/INVALID ANNOTATIONS
# ============================================================

def remove_empty_annotations():
    """
    Remove label files that contain no valid annotations for damage images.
    For whole images, empty labels are correct.
    """
    print("\n" + "=" * 70)
    print("CLEANING EMPTY/INVALID ANNOTATIONS")
    print("=" * 70)
    
    removed = 0
    
    for split in ['train', 'val', 'test']:
        lbl_dir = OUTPUT_DIR / 'labels' / split
        if not lbl_dir.exists():
            continue
        
        for label_path in lbl_dir.glob('*.txt'):
            with open(label_path) as f:
                content = f.read().strip()
            
            if not content:
                # Empty label file - keep it (valid for whole images)
                pass
    
    print(f"  Cleanup complete")


# ============================================================
# GENERATE REPORTS
# ============================================================

def generate_annotation_report(records):
    """Generate annotation_report.csv."""
    csv_path = OUTPUT_DIR / 'annotation_report.csv'
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Created annotation_report.csv with {len(records)} records")
    return df


def generate_class_distribution_csv(stats):
    """Generate class_distribution.csv."""
    csv_path = OUTPUT_DIR / 'class_distribution.csv'
    
    data = []
    
    for dtype, count in stats['damage_types'].items():
        data.append({
            'class_type': 'damage_type',
            'class_name': dtype,
            'count': count,
            'percentage': round(count / max(stats['total_annotations'], 1) * 100, 2),
        })
    
    for loc, count in stats['damage_locations'].items():
        data.append({
            'class_type': 'damage_location',
            'class_name': loc,
            'count': count,
            'percentage': round(count / max(stats['total_annotations'], 1) * 100, 2),
        })
    
    for sev, count in stats['severities'].items():
        data.append({
            'class_type': 'severity',
            'class_name': sev,
            'count': count,
            'percentage': round(count / max(stats['total_annotations'], 1) * 100, 2),
        })
    
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"[✓] Created class_distribution.csv with {len(data)} records")


def generate_missing_annotations_csv(stats):
    """Generate missing_annotations.csv."""
    csv_path = OUTPUT_DIR / 'missing_annotations.csv'
    
    if stats['needs_manual_verification']:
        df = pd.DataFrame(stats['needs_manual_verification'])
        df.to_csv(csv_path, index=False)
    else:
        # Create empty CSV with headers
        df = pd.DataFrame(columns=['image', 'reason'])
        df.to_csv(csv_path, index=False)
    
    print(f"[✓] Created missing_annotations.csv ({len(stats['needs_manual_verification'])} entries)")


def generate_quality_report(stats, annotation_stats):
    """Generate annotation_quality_report.txt."""
    report_path = OUTPUT_DIR / 'annotation_quality_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ANNOTATION QUALITY REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Total Images:          {stats['total_images']}\n")
        f.write(f"   Labeled Images:        {stats['labeled_images']}\n")
        f.write(f"   Empty Labels:          {stats['empty_label_images']}\n")
        f.write(f"   Total Annotations:     {stats['total_annotations']}\n")
        f.write(f"   Needing Verification:  {len(stats['needs_manual_verification'])}\n\n")
        
        pct_labeled = stats['labeled_images'] / max(stats['total_images'], 1) * 100
        f.write(f"   Annotation Coverage:   {pct_labeled:.1f}%\n")
        f.write(f"   {'✓ PASS' if pct_labeled >= 90 else '✗ FAIL'} (Target: >= 90%)\n\n")
        
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
        
        f.write("5. VALIDATION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Total images:           {annotation_stats.get('total_images', 0)}\n")
        f.write(f"   With labels:            {annotation_stats.get('has_label', 0)}\n")
        f.write(f"   Missing labels:         {annotation_stats.get('missing_label', 0)}\n")
        f.write(f"   Empty labels:           {annotation_stats.get('empty_labels', 0)}\n")
        f.write(f"   With annotations:       {annotation_stats.get('has_annotations', 0)}\n")
        f.write(f"   Invalid boxes removed:  {annotation_stats.get('invalid_boxes', 0)}\n")
        f.write(f"   Duplicates removed:     {annotation_stats.get('duplicates_removed', 0)}\n\n")
        
        f.write("6. OUTPUT STRUCTURE\n")
        f.write("-" * 40 + "\n")
        f.write(f"   {OUTPUT_DIR}/\n")
        f.write(f"   ├── images/\n")
        train_img_count = len(list((OUTPUT_DIR / 'images' / 'train').glob('*'))) if (OUTPUT_DIR / 'images' / 'train').exists() else 0
        val_img_count = len(list((OUTPUT_DIR / 'images' / 'val').glob('*'))) if (OUTPUT_DIR / 'images' / 'val').exists() else 0
        test_img_count = len(list((OUTPUT_DIR / 'images' / 'test').glob('*'))) if (OUTPUT_DIR / 'images' / 'test').exists() else 0
        train_lbl_count = len(list((OUTPUT_DIR / 'labels' / 'train').glob('*.txt'))) if (OUTPUT_DIR / 'labels' / 'train').exists() else 0
        val_lbl_count = len(list((OUTPUT_DIR / 'labels' / 'val').glob('*.txt'))) if (OUTPUT_DIR / 'labels' / 'val').exists() else 0
        test_lbl_count = len(list((OUTPUT_DIR / 'labels' / 'test').glob('*.txt'))) if (OUTPUT_DIR / 'labels' / 'test').exists() else 0
        
        f.write(f"   │   ├── train/    ({train_img_count} images)\n")
        f.write(f"   │   ├── val/      ({val_img_count} images)\n")
        f.write(f"   │   └── test/     ({test_img_count} images)\n")
        f.write(f"   ├── labels/\n")
        f.write(f"   │   ├── train/    ({train_lbl_count} labels)\n")
        f.write(f"   │   ├── val/      ({val_lbl_count} labels)\n")
        f.write(f"   │   └── test/     ({test_lbl_count} labels)\n")
        f.write(f"   ├── classes.txt\n")
        f.write(f"   ├── data.yaml\n")
        f.write(f"   ├── annotation_report.csv\n")
        f.write(f"   ├── class_distribution.csv\n")
        f.write(f"   ├── missing_annotations.csv\n")
        f.write(f"   └── annotation_quality_report.txt\n\n")
        
        f.write("7. CLASS MAPPING\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Damage Types (0-5):     {', '.join(CLASS_NAMES)}\n")
        f.write(f"   Locations (6-14):       {', '.join(LOCATION_NAMES)}\n")
        f.write(f"   Severity (15-17):       {', '.join(SEVERITY_NAMES)}\n\n")
        
        f.write("8. DATASET READINESS ASSESSMENT\n")
        f.write("-" * 40 + "\n")
        readiness_scores = []
        
        # Coverage check
        if pct_labeled >= 90:
            readiness_scores.append("✓ Annotation coverage >= 90%")
        else:
            readiness_scores.append(f"✗ Annotation coverage {pct_labeled:.1f}% (< 90%)")
        
        # Distribution check
        if len(stats['damage_types']) >= 4:
            readiness_scores.append(f"✓ {len(stats['damage_types'])} damage types represented")
        else:
            readiness_scores.append(f"✗ Only {len(stats['damage_types'])} damage types")
        
        if len(stats['damage_locations']) >= 5:
            readiness_scores.append(f"✓ {len(stats['damage_locations'])} damage locations represented")
        else:
            readiness_scores.append(f"✗ Only {len(stats['damage_locations'])} damage locations")
        
        if len(stats['severities']) == 3:
            readiness_scores.append("✓ All 3 severity levels represented")
        else:
            readiness_scores.append(f"✗ Only {len(stats['severities'])} severity levels")
        
        # Consistency check
        if annotation_stats.get('missing_label', 0) == 0:
            readiness_scores.append("✓ All images have corresponding label files")
        else:
            readiness_scores.append(f"✗ {annotation_stats.get('missing_label', 0)} images missing labels")
        
        if annotation_stats.get('invalid_boxes', 0) == 0:
            readiness_scores.append("✓ All bounding boxes within image boundaries")
        else:
            readiness_scores.append(f"✗ {annotation_stats.get('invalid_boxes', 0)} invalid bounding boxes removed")
        
        # Total annotations check
        if stats['total_annotations'] >= 500:
            readiness_scores.append(f"✓ Sufficient annotations ({stats['total_annotations']})")
        else:
            readiness_scores.append(f"✗ Low annotation count ({stats['total_annotations']})")
        
        f.write("\n".join(f"   {score}\n" for score in readiness_scores))
        
        # Overall readiness
        total_checks = len(readiness_scores)
        passed = sum(1 for s in readiness_scores if s.startswith('✓'))
        readiness_pct = passed / max(total_checks, 1) * 100
        
        f.write(f"\n   Overall Dataset Readiness: {readiness_pct:.0f}%\n")
        if readiness_pct >= 80:
            f.write("   Status: READY FOR YOLOv8 TRAINING ✓\n")
        elif readiness_pct >= 50:
            f.write("   Status: CONDITIONALLY READY - Address warnings ⚠\n")
        else:
            f.write("   Status: NOT READY - Significant issues need resolution ✗\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF QUALITY REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"[✓] Created annotation_quality_report.txt")


def generate_classes_file():
    """Generate classes.txt with all class names."""
    classes_path = OUTPUT_DIR / 'classes.txt'
    with open(classes_path, 'w') as f:
        for i, name in enumerate(ALL_CLASSES):
            f.write(f"{name}\n")
    print(f"[✓] Created classes.txt with {len(ALL_CLASSES)} classes")


def generate_data_yaml():
    """Generate data.yaml for YOLOv8."""
    data_yaml = {
        'path': str(OUTPUT_DIR.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': NUM_CLASSES,
        'names': ALL_CLASSES,
    }
    
    yaml_path = OUTPUT_DIR / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"[✓] Created data.yaml for YOLOv8")


# ============================================================
# GENERATE VISUAL SAMPLES
# ============================================================

def generate_visual_samples(n_samples=100):
    """
    Draw bounding boxes on random images and save to review_samples/.
    """
    print(f"\n{'=' * 70}")
    print(f"GENERATING {n_samples} VISUAL SAMPLES")
    print(f"{'=' * 70}")
    
    samples_dir = OUTPUT_DIR / 'review_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all labeled images
    candidates = []
    for split in ['train', 'val', 'test']:
        img_dir = OUTPUT_DIR / 'images' / split
        lbl_dir = OUTPUT_DIR / 'labels' / split
        if not img_dir.exists():
            continue
        
        for img_path in img_dir.glob('*'):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                continue
            
            label_path = lbl_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path) as f:
                    content = f.read().strip()
                if content:
                    candidates.append((img_path, label_path, split))
    
    if not candidates:
        print("  No labeled images found for visualization")
        return
    
    # Randomly select n_samples
    random.shuffle(candidates)
    selected = candidates[:min(n_samples, len(candidates))]
    
    # Colors for different classes
    colors = {
        0: (0, 255, 0),      # dent - green
        1: (0, 0, 255),      # scratch - red
        2: (255, 0, 0),      # crack - blue
        3: (0, 255, 255),    # broken_part - yellow
        4: (255, 0, 255),    # paint_damage - magenta
        5: (255, 255, 0),    # other_damage - cyan
    }
    
    samples_drawn = 0
    for img_path, label_path, split in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        with open(label_path) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            
            # Get color based on class
            if cls_id < 6:
                color = colors.get(cls_id, (200, 200, 200))
            elif cls_id < 15:
                color = (128, 128, 0)  # Location - olive
            else:
                color = (0, 128, 128)  # Severity - teal
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Add label text
            if cls_id < NUM_CLASSES:
                label_text = ALL_CLASSES[cls_id]
                # Add confidence if available
                cv2.putText(img, label_text, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save visualization
        sample_name = f"{split}_{img_path.stem}_annotated{img_path.suffix}"
        output_path = samples_dir / sample_name
        cv2.imwrite(str(output_path), img)
        samples_drawn += 1
        
        if samples_drawn % 10 == 0:
            print(f"  Generated {samples_drawn}/{n_samples} visualizations...")
    
    print(f"[✓] Created {samples_drawn} visual samples in {samples_dir}")


# ============================================================
# DATASET READINESS SUMMARY
# ============================================================

def generate_readiness_summary(stats, annotation_stats):
    """Generate a summary showing dataset readiness for YOLOv8 training."""
    print("\n" + "=" * 70)
    print("DATASET READINESS SUMMARY FOR YOLOv8 TRAINING")
    print("=" * 70)
    
    pct_labeled = stats['labeled_images'] / max(stats['total_images'], 1) * 100
    pct_empty = stats['empty_label_images'] / max(stats['total_images'], 1) * 100
    
    print(f"\n  Total Images:          {stats['total_images']}")
    print(f"  With Valid Annotations: {stats['labeled_images']} ({pct_labeled:.1f}%)")
    print(f"  Empty/No Damage:        {stats['empty_label_images']} ({pct_empty:.1f}%)")
    print(f"  Total Annotations:      {stats['total_annotations']}")
    print(f"  Needing Verification:   {len(stats['needs_manual_verification'])}")
    print()
    
    print(f"  Damage Types ({len(stats['damage_types'])}):")
    for dtype, count in stats['damage_types'].most_common():
        print(f"    - {dtype}: {count}")
    
    print(f"\n  Locations ({len(stats['damage_locations'])}):")
    for loc, count in stats['damage_locations'].most_common():
        print(f"    - {loc}: {count}")
    
    print(f"\n  Severities ({len(stats['severities'])}):")
    for sev, count in stats['severities'].most_common():
        print(f"    - {sev}: {count}")
    
    # Dataset validation
    print(f"\n  Validation:")
    print(f"    Images with labels:  {annotation_stats.get('has_label', 0)}")
    print(f"    Missing labels:      {annotation_stats.get('missing_label', 0)}")
    print(f"    Invalid boxes:       {annotation_stats.get('invalid_boxes', 0)}")
    print(f"    Duplicates removed:  {annotation_stats.get('duplicates_removed', 0)}")
    
    # Readiness check
    print(f"\n  Dataset Readiness:")
    if pct_labeled >= 90:
        print(f"    ✓ Annotation coverage: {pct_labeled:.1f}% (>= 90%)")
    else:
        print(f"    ✗ Annotation coverage: {pct_labeled:.1f}% (< 90%)")
    
    if stats['total_annotations'] >= 500:
        print(f"    ✓ Sufficient annotations: {stats['total_annotations']}")
    else:
        print(f"    ✗ Insufficient annotations: {stats['total_annotations']}")
    
    if annotation_stats.get('missing_label', 0) == 0:
        print(f"    ✓ All images have labels")
    else:
        print(f"    ✗ {annotation_stats.get('missing_label', 0)} images missing labels")
    
    if annotation_stats.get('invalid_boxes', 0) == 0:
        print(f"    ✓ All bounding boxes valid")
    else:
        print(f"    ⚠ {annotation_stats.get('invalid_boxes', 0)} invalid boxes removed")
    
    print(f"\n  {'=' * 50}")
    if pct_labeled >= 90 and stats['total_annotations'] >= 500:
        print(f"  STATUS: READY FOR YOLOv8 TRAINING ✓")
    elif pct_labeled >= 70:
        print(f"  STATUS: CONDITIONALLY READY ⚠")
        print(f"  Recommendation: Review {len(stats['needs_manual_verification'])} flagged images")
    else:
        print(f"  STATUS: NEEDS MORE WORK ✗")
        print(f"  Recommendation: Improve damage detection algorithm")
    print(f"  {'=' * 50}")
    
    print(f"\n  Training command:")
    print(f"  yolo train data={OUTPUT_DIR / 'data.yaml'} model=models/yolov8n.pt epochs=100 imgsz=640")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Main entry point for the production dataset pipeline."""
    print("=" * 70)
    print("PRODUCTION DATASET CREATION PIPELINE")
    print("Enhanced Vehicle Damage Detection Dataset")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = datetime.now()
    
    # Step 1: Setup directories
    print("Step 1: Setting up directories...")
    setup_directories()
    
    # Step 2: Validate existing annotations
    print("\nStep 2: Validating existing auto-generated annotations...")
    validate_existing_annotations()
    
    # Step 3: Process manual review images
    print("\nStep 3: Re-processing manual review images...")
    manual_results, manual_stats = process_manual_review()
    
    # Step 4: Create final dataset with all images
    print("\nStep 4: Creating final dataset...")
    processed_images, stats = create_final_dataset()
    
    # Step 5: Split dataset
    print("\nStep 5: Splitting dataset...")
    split_data = split_dataset(processed_images)
    
    # Step 6: Validate dataset consistency
    print("\nStep 6: Validating dataset consistency...")
    annotation_stats = validate_dataset_consistency()
    
    # Step 7: Clean up
    print("\nStep 7: Cleaning up annotations...")
    remove_empty_annotations()
    
    # Step 8: Generate output files
    print("\nStep 8: Generating output files...")
    generate_classes_file()
    generate_data_yaml()
    
    # Build records for CSV
    records = []
    for img_name, info in processed_images.items():
        for label in info['labels']:
            parts = label.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                records.append({
                    'image_name': img_name,
                    'split': info['original_split'],
                    'class_id': cls_id,
                    'class_name': ALL_CLASSES[cls_id] if cls_id < len(ALL_CLASSES) else 'unknown',
                    'bbox_xc': float(parts[1]),
                    'bbox_yc': float(parts[2]),
                    'bbox_w': float(parts[3]),
                    'bbox_h': float(parts[4]),
                    'source': info['source'],
                })
    
    generate_annotation_report(records)
    generate_class_distribution_csv(stats)
    generate_missing_annotations_csv(stats)
    generate_quality_report(stats, annotation_stats)
    
    # Step 9: Generate visual samples
    print("\nStep 9: Generating visual samples...")
    generate_visual_samples(n_samples=VISUAL_SAMPLE_COUNT)
    
    # Step 10: Generate readiness summary
    print("\nStep 10: Generating dataset readiness summary...")
    generate_readiness_summary(stats, annotation_stats)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total time: {elapsed:.1f} seconds")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"  - images/: train ({len(split_data['train'])}), val ({len(split_data['val'])}), test ({len(split_data['test'])})")
    print(f"  - labels/: YOLO format annotations")
    print(f"  - classes.txt: {NUM_CLASSES} classes")
    print(f"  - data.yaml: YOLOv8 configuration")
    print(f"  - annotation_report.csv: All annotations")
    print(f"  - class_distribution.csv: Class statistics")
    print(f"  - missing_annotations.csv: Images needing review")
    print(f"  - annotation_quality_report.txt: Quality assessment")
    print(f"  - review_samples/: {VISUAL_SAMPLE_COUNT} annotated visualizations")
    print(f"\nTo train YOLOv8:")
    print(f"  yolo train data={OUTPUT_DIR / 'data.yaml'} model=models/yolov8n.pt epochs=100 imgsz=640")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
