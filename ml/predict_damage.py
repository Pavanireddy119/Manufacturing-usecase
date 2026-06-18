"""
Vehicle Damage Detection - YOLOv8 Prediction & Damage Report Script
Runs inference on test images and generates structured damage reports.
Includes comprehensive debugging and diagnostics.
"""

import os
import sys
import json
import yaml
import torch
import random
import numpy as np
import traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# ===================== CONFIGURATION =====================

CLASS_NAMES = [
    # Damage Types (0-5)
    'dent', 'scratch', 'crack', 'broken_part', 'paint_damage', 'other_damage',
    # Locations (6-14)
    'front_bumper', 'rear_bumper', 'hood', 'windshield', 'left_door', 'right_door',
    'roof', 'side_panel', 'other_location',
    # Severity (15-17)
    'low', 'medium', 'high'
]

DAMAGE_TYPES = {
    0: 'dent', 1: 'scratch', 2: 'crack', 3: 'broken_part',
    4: 'paint_damage', 5: 'other_damage'
}

DAMAGE_LOCATIONS = {
    6: 'front_bumper', 7: 'rear_bumper', 8: 'hood', 9: 'windshield',
    10: 'left_door', 11: 'right_door', 12: 'roof', 13: 'side_panel',
    14: 'other_location'
}

SEVERITY_LEVELS = {
    15: 'low', 16: 'medium', 17: 'high'
}

# Group indices
DAMAGE_GROUP = list(range(0, 6))
LOCATION_GROUP = list(range(6, 15))
SEVERITY_GROUP = list(range(15, 18))

# Colors for visualization
COLORS = {
    'damage': '#FF4444',
    'location': '#4488FF',
    'severity': '#44CC44',
    'box': '#FF6B35',
    'text_bg': 'rgba(0,0,0,0.7)',
}

# Matplotlib-compatible colors
MPL_COLORS = {
    0: '#FF4444', 1: '#FF8844', 2: '#FFAA44', 3: '#FFCC44',
    4: '#44FF44', 5: '#44FFAA',  # damage types
    6: '#4444FF', 7: '#4488FF', 8: '#44AAFF', 9: '#44CCFF',
    10: '#8844FF', 11: '#AA44FF', 12: '#CC44FF', 13: '#FF44FF',
    14: '#FF44CC',  # locations
    15: '#44FF44', 16: '#FFAA00', 17: '#FF4444',  # severity
}


def find_model_auto():
    """Automatically locate a YOLOv8 .pt model file."""
    search_paths = [
        'output/models/best.pt',
        'output/models/last.pt',
        'output/models/weights/best.pt',
        'output/models/weights/last.pt',
        'runs/detect/train/weights/best.pt',
        'runs/detect/train/weights/last.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train2/weights/last.pt',
        'best.pt',
        'last.pt',
        'yolov8s.pt',
        'yolov8n.pt',
    ]
    for p in search_paths:
        if os.path.exists(p):
            print(f"  [AUTO-LOCATE] Found model at: {p}")
            return p
    
    # Recursive search as fallback
    print("  [AUTO-LOCATE] Searching recursively for *.pt files...")
    import glob
    pt_files = glob.glob('**/*.pt', recursive=True)
    # Prefer best.pt
    best_files = [f for f in pt_files if 'best' in f.lower()]
    if best_files:
        print(f"  [AUTO-LOCATE] Found: {best_files[0]}")
        return best_files[0]
    if pt_files:
        print(f"  [AUTO-LOCATE] Found: {pt_files[0]}")
        return pt_files[0]
    return None


def load_model(model_path=None):
    """Load trained YOLOv8 model with auto-locate fallback."""
    
    # If no path given, auto-locate
    if model_path is None or not os.path.exists(model_path):
        print("[DEBUG] Model path not found or not specified. Auto-locating...")
        model_path = find_model_auto()
        if model_path is None:
            raise FileNotFoundError("No YOLOv8 model (.pt) found anywhere in the project!")
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"[DEBUG] Model path loaded: {model_path.resolve()}")
    print(f"[DEBUG] Model file size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    
    try:
        model = YOLO(str(model_path))
        print("[DEBUG] Model loaded successfully from ultralytics.")
        
        # Print model classes
        print(f"\n[DEBUG] Model classes loaded from {model_path.name}:")
        if hasattr(model, 'names'):
            for cls_id, cls_name in model.names.items():
                print(f"  Class {cls_id}: {cls_name}")
        else:
            print("  (model.names not available)")
        
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()
        raise


def group_detections(results):
    """
    Group YOLO detection results into damage instances.
    Each damage instance has: damage_type, location, severity with same bbox.
    """
    instances = []
    
    if results is None or len(results) == 0:
        return instances
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return instances
    
    boxes = result.boxes
    
    print(f"  [DEBUG] Raw detections from model: {len(boxes)} boxes")
    
    # Extract all detections
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'class_{cls_id}'
        print(f"  [DEBUG] Detection {i}: class_id={cls_id} ({class_name}), conf={conf:.4f}, bbox={[round(x,2) for x in xyxy]}")
        
        detections.append({
            'class_id': cls_id,
            'class_name': class_name,
            'confidence': conf,
            'bbox': xyxy,
        })
    
    # Group detections by bounding box proximity (same damage instance)
    used_indices = set()
    
    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue
        
        instance = {
            'damage_type': None,
            'damage_location': None,
            'severity': None,
            'confidence': 0.0,
            'bounding_box': det1['bbox'],
        }
        
        used_indices.add(i)
        
        # Find matching detections (same bbox)
        for j, det2 in enumerate(detections):
            if j in used_indices:
                continue
            
            # Check if bboxes overlap significantly (IoU > 0.5)
            if bbox_iou(det1['bbox'], det2['bbox']) > 0.5:
                used_indices.add(j)
                
                cls_id = det2['class_id']
                if cls_id in DAMAGE_GROUP:
                    instance['damage_type'] = DAMAGE_TYPES.get(cls_id, 'unknown')
                elif cls_id in LOCATION_GROUP:
                    instance['damage_location'] = DAMAGE_LOCATIONS.get(cls_id, 'unknown')
                elif cls_id in SEVERITY_GROUP:
                    instance['severity'] = SEVERITY_LEVELS.get(cls_id, 'unknown')
                
                instance['confidence'] = max(instance['confidence'], det2['confidence'])
        
        # Fill in from current detection if not already set
        cls_id = det1['class_id']
        if cls_id in DAMAGE_GROUP and instance['damage_type'] is None:
            instance['damage_type'] = DAMAGE_TYPES.get(cls_id, 'unknown')
        elif cls_id in LOCATION_GROUP and instance['damage_location'] is None:
            instance['damage_location'] = DAMAGE_LOCATIONS.get(cls_id, 'unknown')
        elif cls_id in SEVERITY_GROUP and instance['severity'] is None:
            instance['severity'] = SEVERITY_LEVELS.get(cls_id, 'unknown')
        
        instance['confidence'] = max(instance['confidence'], det1['confidence'])
        
        instances.append(instance)
    
    print(f"  [DEBUG] Grouped into {len(instances)} damage instances")
    return instances


def bbox_iou(box1, box2):
    """Calculate IoU of two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def predict_image(model, image_path, conf_threshold=0.25, iou_threshold=0.5):
    """Run inference on a single image."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"  [DEBUG] Running prediction on: {image_path.name} (conf={conf_threshold}, iou={iou_threshold})")
    
    # Run prediction
    try:
        results = model(
            str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            verbose=False,
        )
        print(f"  [DEBUG] Prediction completed. Results type: {type(results).__name__}")
        print(f"  [DEBUG] Number of results returned: {len(results) if results else 0}")
        return results
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return None


def visualize_predictions(image_path, instances, output_path, class_name='all'):
    """Visualize predictions on the image."""
    from matplotlib.patches import FancyBboxPatch
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)
        
        # Draw each instance
        for inst in instances:
            bbox = inst['bounding_box']
            x1, y1, x2, y2 = bbox
            
            # Create label
            label_parts = []
            if inst['damage_type']:
                label_parts.append(f"Type: {inst['damage_type']}")
            if inst['damage_location']:
                label_parts.append(f"Loc: {inst['damage_location']}")
            if inst['severity']:
                label_parts.append(f"Sev: {inst['severity']}")
            label_parts.append(f"Conf: {inst['confidence']:.2f}")
            label = '\n'.join(label_parts)
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='#FF6B35', facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Draw label background
            label_bg = patches.FancyBboxPatch(
                (x1, y1 - 60), 120, 60,
                boxstyle="round,pad=0.3",
                facecolor='black', alpha=0.7, edgecolor='none'
            )
            ax.add_patch(label_bg)
            
            # Add label text
            ax.text(
                x1 + 5, y1 - 50, label,
                fontsize=7, color='white', va='top', ha='left',
                fontfamily='monospace',
            )
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='#FF6B35', edgecolor='#FF6B35', label='Damage Box'),
            patches.Patch(facecolor='#44FF44', edgecolor='#44FF44', label='Low Severity'),
            patches.Patch(facecolor='#FFAA00', edgecolor='#FFAA00', label='Medium Severity'),
            patches.Patch(facecolor='#FF4444', edgecolor='#FF4444', label='High Severity'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        ax.axis('off')
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  [DEBUG] Saved prediction visualization: {output_path}")
        print(f"  [DEBUG] Output file exists: {os.path.exists(output_path)}")
        return True
    except Exception as e:
        print(f"  [ERROR] Visualization failed: {e}")
        traceback.print_exc()
        return False


def generate_damage_report(instances, image_path):
    """Generate structured damage report in the required format."""
    report = []
    
    for inst in instances:
        bbox = inst['bounding_box']
        report.append({
            "damage_type": inst.get('damage_type', 'unknown'),
            "damage_location": inst.get('damage_location', 'unknown'),
            "severity": inst.get('severity', 'unknown'),
            "confidence": round(inst.get('confidence', 0.0), 4),
            "bounding_box": [round(bbox[0], 2), round(bbox[1], 2), 
                           round(bbox[2], 2), round(bbox[3], 2)],
        })
    
    return report


def run_inference_on_test_set(model, num_images=20, output_dir='output/predictions',
                              conf_threshold=0.25, data_yaml='dataset_final/data.yaml'):
    """Run inference on random test set images."""
    print("\n" + "="*60)
    print("INFERENCE TESTING ON TEST SET")
    print("="*60)
    
    # Load data config
    try:
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        print(f"[DEBUG] Data YAML loaded: {data_yaml}")
        print(f"[DEBUG] Data config: {data_cfg}")
    except Exception as e:
        print(f"[ERROR] Failed to load data YAML: {e}")
        traceback.print_exc()
        return {}
    
    base_path = Path(data_cfg.get('path', '.')).resolve()
    test_img_path = base_path / 'images' / 'test'
    
    print(f"[DEBUG] Test images path from YAML: {test_img_path}")
    print(f"[DEBUG] Test images path exists: {test_img_path.exists()}")
    
    if not test_img_path.exists():
        print(f"[ERROR] Test images path not found: {test_img_path}")
        print(f"[DEBUG] Listing contents of {base_path / 'images'}:")
        images_dir = base_path / 'images'
        if images_dir.exists():
            for d in images_dir.iterdir():
                print(f"  {d.name}/ (dir={d.is_dir()})")
        return {}
    
    # Get test images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    test_images = [f for f in test_img_path.iterdir() if f.suffix in image_extensions]
    
    print(f"[DEBUG] Number of test images found: {len(test_images)}")
    
    if not test_images:
        print("[ERROR] No test images found.")
        return {}
    
    # Select random images
    num_images = min(num_images, len(test_images))
    selected_images = random.sample(test_images, num_images)
    
    print(f"[DEBUG] Selected {num_images} images for inference")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DEBUG] Output directory created: {output_dir.resolve()}")
    print(f"[DEBUG] Output dir exists: {output_dir.exists()}")
    
    all_reports = {}
    
    for i, img_path in enumerate(selected_images):
        print(f"\n[{i+1}/{num_images}] Processing: {img_path.name}")
        
        # Run prediction
        try:
            results = predict_image(model, str(img_path), conf_threshold)
        except Exception as e:
            print(f"  [ERROR] Prediction failed for {img_path.name}: {e}")
            traceback.print_exc()
            continue
        
        # Group detections into damage instances
        try:
            instances = group_detections(results)
        except Exception as e:
            print(f"  [ERROR] Grouping failed for {img_path.name}: {e}")
            traceback.print_exc()
            continue
        
        # Generate report
        try:
            report = generate_damage_report(instances, str(img_path))
            all_reports[img_path.name] = report
        except Exception as e:
            print(f"  [ERROR] Report generation failed for {img_path.name}: {e}")
            traceback.print_exc()
            continue
        
        print(f"  [DEBUG] {len(instances)} damage instances detected")
        
        # Visualize predictions
        try:
            output_img_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
            print(f"  [DEBUG] Saving prediction to: {output_img_path}")
            vis_success = visualize_predictions(str(img_path), instances, str(output_img_path))
            print(f"  [DEBUG] Visualization saved: {vis_success}")
        except Exception as e:
            print(f"  [ERROR] Visualization save failed: {e}")
            traceback.print_exc()
        
        # Print damage report in required format
        if report:
            print(f"  [REPORT] Damage Summary for {img_path.name}:")
            print(f"  {'Damage Type':20s} {'Damage Location':20s} {'Severity':12s} {'Confidence':12s} {'Bounding Box':30s}")
            print(f"  {'-'*20} {'-'*20} {'-'*12} {'-'*12} {'-'*30}")
            for j, det in enumerate(report):
                dtype = str(det['damage_type']) if det['damage_type'] is not None else 'unknown'
                dloc = str(det['damage_location']) if det['damage_location'] is not None else 'unknown'
                sev = str(det['severity']) if det['severity'] is not None else 'unknown'
                bbox_str = f"[{det['bounding_box'][0]:.0f},{det['bounding_box'][1]:.0f},{det['bounding_box'][2]:.0f},{det['bounding_box'][3]:.0f}]"
                print(f"  {dtype:20s} {dloc:20s} {sev:12s} {det['confidence']:<12.4f} {bbox_str:30s}")
        else:
            print(f"  [INFO] No damage detected in {img_path.name}")
    
    # Save full report as JSON
    if all_reports:
        try:
            report_path = output_dir / 'damage_reports.json'
            print(f"\n[DEBUG] Saving damage reports JSON to: {report_path}")
            with open(report_path, 'w') as f:
                json.dump(all_reports, f, indent=2)
            print(f"[DEBUG] JSON report saved: {os.path.exists(report_path)}")
            print(f"[DEBUG] JSON file size: {os.path.getsize(report_path)} bytes")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON report: {e}")
            traceback.print_exc()
    else:
        print("[WARNING] No reports generated - all images had errors or no detections")
        # Still create an empty report
        try:
            report_path = output_dir / 'damage_reports.json'
            with open(report_path, 'w') as f:
                json.dump({"info": "No damages detected"}, f, indent=2)
            print(f"[INFO] Created empty report at {report_path}")
        except Exception as e:
            print(f"[ERROR] Failed to create empty report: {e}")
    
    print(f"\n[DEBUG] Total images processed: {len(all_reports)}")
    print(f"[DEBUG] Output directory contents:")
    for f in output_dir.iterdir():
        print(f"  {f.name} (size: {os.path.getsize(f)} bytes)")
    
    return all_reports


def main():
    """Main prediction pipeline."""
    print("="*60)
    print("VEHICLE DAMAGE DETECTION - PREDICTION PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Vehicle Damage Detection Inference')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (auto-locates if not specified)')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to single image or directory')
    parser.add_argument('--output', type=str, default='output/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--num-test', type=int, default=20,
                       help='Number of test images to process')
    parser.add_argument('--data-yaml', type=str, default='dataset_final/data.yaml',
                       help='Dataset config file')
    
    args = parser.parse_args()
    
    print(f"\n[DEBUG] Arguments:")
    print(f"  Model path: {args.model or 'AUTO-LOCATE'}")
    print(f"  Source: {args.source or 'test set'}")
    print(f"  Output: {args.output}")
    print(f"  Confidence threshold: {args.conf}")
    print(f"  Num test images: {args.num_test}")
    print(f"  Data YAML: {args.data_yaml}")
    
    # Load model
    try:
        model = load_model(args.model)
    except FileNotFoundError as e:
        print(f"  [FATAL] {e}")
        return
    except Exception as e:
        print(f"  [FATAL] Model loading failed: {e}")
        traceback.print_exc()
        return
    
    if args.source:
        # Single image or directory inference
        source_path = Path(args.source)
        if source_path.is_file():
            print(f"\nProcessing single image: {source_path}")
            try:
                results = predict_image(model, str(source_path), args.conf)
                instances = group_detections(results)
                report = generate_damage_report(instances, str(source_path))
                
                print(f"\nDamage Report:")
                print(json.dumps(report, indent=2))
                
                # Print in required format
                if report:
                    print(f"\n{'Damage Type':20s} {'Damage Location':20s} {'Severity':12s} {'Confidence':12s} {'Bounding Box':30s}")
                    print(f"{'-'*20} {'-'*20} {'-'*12} {'-'*12} {'-'*30}")
                    for det in report:
                        dtype = str(det['damage_type']) if det['damage_type'] is not None else 'unknown'
                        dloc = str(det['damage_location']) if det['damage_location'] is not None else 'unknown'
                        sev = str(det['severity']) if det['severity'] is not None else 'unknown'
                        bbox_str = f"[{det['bounding_box'][0]:.0f},{det['bounding_box'][1]:.0f},{det['bounding_box'][2]:.0f},{det['bounding_box'][3]:.0f}]"
                        print(f"{dtype:20s} {dloc:20s} {sev:12s} {det['confidence']:<12.4f} {bbox_str:30s}")
                
                # Visualize
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)
                output_img_path = output_path / f"{source_path.stem}_pred{source_path.suffix}"
                visualize_predictions(str(source_path), instances, str(output_img_path))
                print(f"\nVisualization saved to: {output_img_path}")
            except Exception as e:
                print(f"[ERROR] Single image processing failed: {e}")
                traceback.print_exc()
        elif source_path.is_dir():
            # Process all images in directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
            images = [f for f in source_path.iterdir() if f.suffix in image_extensions]
            print(f"\nProcessing {len(images)} images from: {source_path}")
            
            all_reports = {}
            for img_path in images:
                try:
                    results = predict_image(model, str(img_path), args.conf)
                    instances = group_detections(results)
                    report = generate_damage_report(instances, str(img_path))
                    all_reports[img_path.name] = report
                    
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    output_img_path = output_path / f"{img_path.stem}_pred{img_path.suffix}"
                    visualize_predictions(str(img_path), instances, str(output_img_path))
                except Exception as e:
                    print(f"  [ERROR] Processing {img_path.name} failed: {e}")
                    traceback.print_exc()
            
            # Save reports
            report_path = Path(args.output) / 'damage_reports.json'
            try:
                with open(report_path, 'w') as f:
                    json.dump(all_reports, f, indent=2)
                print(f"\nReports saved to: {report_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save reports: {e}")
                traceback.print_exc()
    else:
        # Run on test set
        run_inference_on_test_set(
            model, 
            num_images=args.num_test,
            output_dir=args.output,
            conf_threshold=args.conf,
            data_yaml=args.data_yaml,
        )
    
    # Final output listing
    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"\n{'='*60}")
        print(f"OUTPUT DIRECTORY: {output_dir.resolve()}")
        print(f"{'='*60}")
        files = list(output_dir.iterdir())
        if files:
            for f in files:
                print(f"  {f.name} ({os.path.getsize(f)} bytes)")
        else:
            print(f"  [WARNING] Output directory is EMPTY!")
    
    print(f"\n{'='*60}")
    print(f"PREDICTION COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()