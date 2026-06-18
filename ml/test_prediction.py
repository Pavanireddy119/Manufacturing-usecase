"""
test_prediction.py - Quick YOLOv8 Prediction Test Script
Loads the best.pt model, selects one test image, runs prediction,
prints detected classes, and saves one prediction image.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

try:
    from ultralytics import YOLO
except ImportError:
    os.system('pip install ultralytics')
    from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def find_model():
    """Find a .pt model file."""
    search_paths = [
        'output/models/best.pt',
        'output/models/weights/best.pt',
        'output/models/last.pt',
        'output/models/weights/last.pt',
        'best.pt',
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    import glob
    pt_files = glob.glob('**/*.pt', recursive=True)
    best = [f for f in pt_files if 'best' in f.lower()]
    return best[0] if best else (pt_files[0] if pt_files else None)


def test_prediction():
    """Run a quick test prediction."""
    print("=" * 60)
    print("TEST PREDICTION SCRIPT")
    print("=" * 60)
    
    # Step 1: Find and load model
    print("\n[Step 1] Finding YOLOv8 model...")
    model_path = find_model()
    if model_path is None:
        print("  ERROR: No .pt model file found!")
        return False
    print(f"  Model found: {model_path}")
    print(f"  Model size: {os.path.getsize(model_path) / 1e6:.2f} MB")
    
    try:
        model = YOLO(model_path)
        print("  Model loaded successfully.")
        
        # Print classes
        print("\n  Model classes:")
        if hasattr(model, 'names'):
            for cls_id, cls_name in model.names.items():
                print(f"    Class {cls_id}: {cls_name}")
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: Find test image
    print("\n[Step 2] Finding test image...")
    test_img_dir = 'dataset_final/images/test'
    if not os.path.exists(test_img_dir):
        test_img_dir = 'dataset_final/images/val'
    if not os.path.exists(test_img_dir):
        print("  ERROR: No test/val image directory found!")
        return False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    test_images = [f for f in Path(test_img_dir).iterdir() if f.suffix in image_extensions]
    
    if not test_images:
        print(f"  ERROR: No images found in {test_img_dir}")
        return False
    
    test_img = test_images[0]
    print(f"  Selected: {test_img}")
    print(f"  Total test images available: {len(test_images)}")
    
    # Step 3: Run prediction
    print("\n[Step 3] Running prediction...")
    try:
        results = model(
            str(test_img),
            conf=0.25,
            iou=0.5,
            imgsz=640,
            verbose=False,
        )
        print(f"  Prediction returned: {len(results)} result(s)")
    except Exception as e:
        print(f"  ERROR during prediction: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Print detected classes
    print("\n[Step 4] Detections:")
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"  Total boxes detected: {len(result.boxes)}")
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            conf = float(result.boxes.conf[i].item())
            xyxy = result.boxes.xyxy[i].tolist()
            class_name = model.names.get(cls_id, f'class_{cls_id}')
            print(f"\n  Detection {i + 1}:")
            print(f"    Class: {class_name} (ID {cls_id})")
            print(f"    Confidence: {conf:.4f}")
            print(f"    Bounding Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
    else:
        print("  No detections found (confidence may be too high)")
    
    # Step 5: Save prediction image
    print("\n[Step 5] Saving prediction image...")
    try:
        # Create output dir
        output_dir = Path('test_output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize with matplotlib
        from PIL import Image
        img = Image.open(str(test_img)).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)
        
        if result.boxes is not None and len(result.boxes) > 0:
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                conf = float(result.boxes.conf[i].item())
                xyxy = result.boxes.xyxy[i].tolist()
                class_name = model.names.get(cls_id, f'class_{cls_id}')
                
                rect = patches.Rectangle(
                    (xyxy[0], xyxy[1]), xyxy[2] - xyxy[0], xyxy[3] - xyxy[1],
                    linewidth=2, edgecolor='#FF6B35', facecolor='none', alpha=0.8
                )
                ax.add_patch(rect)
                ax.text(xyxy[0], xyxy[1] - 10, f"{class_name} {conf:.2f}",
                       fontsize=10, color='white',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
        
        ax.axis('off')
        output_path = output_dir / f"test_pred_{test_img.stem}.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
    except Exception as e:
        print(f"  ERROR saving visualization: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST PREDICTION COMPLETE")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_prediction()
    sys.exit(0 if success else 1)