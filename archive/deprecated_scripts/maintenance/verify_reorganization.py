#!/usr/bin/env python3
"""
Final Project Structure Verification
Verifies complete reorganization
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def main():
    root = Path(".")
    
    print("=" * 70)
    print("PROJECT REORGANIZATION VERIFICATION REPORT")
    print("=" * 70)
    
    # Define expected structure
    dirs = {
        "integration_package": "Deployment-ready API files",
        "training": "Model training scripts",
        "evaluation": "Model evaluation scripts",
        "preprocessing": "Data preparation scripts",
        "datasets": "Training datasets",
        "models": "Model weights",
        "reports": "Analysis & metrics",
        "docs": "User documentation",
        "outputs": "Training artifacts",
        "archive": "Experimental/old code",
    }
    
    print("\n✓ DIRECTORY STRUCTURE VERIFICATION")
    print("-" * 70)
    for dir_name, purpose in dirs.items():
        dir_path = root / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.rglob("*")))
            print(f"  ✓ {dir_name:20} - {purpose} ({file_count} items)")
        else:
            print(f"  ✗ {dir_name:20} - MISSING!")
    
    # Check key files in integration_package
    print("\n✓ INTEGRATION PACKAGE VERIFICATION")
    print("-" * 70)
    integration_files = [
        "best_damage_model.pt",
        "inspection_api.py",
        "inspection_pipeline.py",
        "classes.txt",
        "data.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    ip_path = root / "integration_package"
    for file in integration_files:
        file_path = ip_path / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024  # KB
            print(f"  ✓ {file:25} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file:25} MISSING!")
    
    # Check key script files
    print("\n✓ TRAINING SCRIPTS VERIFICATION")
    print("-" * 70)
    training_scripts = [
        "train_damage_detection_v1.py",
        "train_fixed_yolo.py",
        "train_model.py",
        "train_yolo.py",
    ]
    
    for script in training_scripts:
        path = root / "training" / script
        if path.exists():
            print(f"  ✓ {script}")
        else:
            # Check if in root (should be moved)
            root_path = root / script
            if root_path.exists():
                print(f"  ⚠ {script} - STILL IN ROOT (should move to training/)")
    
    print("\n✓ EVALUATION SCRIPTS VERIFICATION")
    print("-" * 70)
    eval_scripts = [
        "evaluate_model.py",
        "evaluate_model_comprehensive.py",
        "evaluate_fixed_model.py",
        "evaluate_yolo.py",
        "generate_metrics.py",
        "verify_output.py",
    ]
    
    for script in eval_scripts:
        path = root / "evaluation" / script
        if path.exists():
            print(f"  ✓ {script}")
    
    print("\n✓ PREPROCESSING SCRIPTS VERIFICATION")
    print("-" * 70)
    preproc_scripts = [
        "dataset_cleaner.py",
        "rebuild_dataset.py",
        "resize_images.py",
        "create_train_test.py",
        "transforms.py",
        "restructure_dataset.py",
        "restructure_for_yolo.py",
        "create_labeled_dataset.py",
        "create_production_dataset.py",
    ]
    
    for script in preproc_scripts:
        path = root / "preprocessing" / script
        if path.exists():
            print(f"  ✓ {script}")
    
    # Check datasets
    print("\n✓ DATASETS VERIFICATION")
    print("-" * 70)
    dataset_dirs = [
        "dataset_cleaned",
        "dataset_final",
        "dataset_labeled",
        "dataset_yolo_fixed",
        "dataset_yolo_restructured",
        "preprocessed_dataset",
        "train",
        "test",
        "validation",
    ]
    
    for dataset in dataset_dirs:
        path = root / "datasets" / dataset
        if path.exists():
            print(f"  ✓ {dataset}/")
    
    # Check models
    print("\n✓ MODELS VERIFICATION")
    print("-" * 70)
    models = [
        "best_damage_model.pt",
        "vehicle_damage_model.h5",
        "yolov8n.pt",
        "yolov8s.pt",
    ]
    
    for model in models:
        path = root / "models" / model
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {model:30} ({size:.1f} MB)")
    
    # Check reports
    print("\n✓ REPORTS VERIFICATION")
    print("-" * 70)
    reports_dir = root / "reports"
    if reports_dir.exists():
        report_files = sorted([f.name for f in reports_dir.glob("*") if f.is_file()])
        for report in report_files:
            print(f"  ✓ {report}")
    
    # Check docs
    print("\n✓ DOCUMENTATION VERIFICATION")
    print("-" * 70)
    docs_dir = root / "docs"
    if docs_dir.exists():
        doc_files = sorted([f.name for f in docs_dir.glob("*") if f.is_file()])
        for doc in doc_files:
            print(f"  ✓ {doc}")
    
    # Check for remaining files in root
    print("\n✓ ROOT DIRECTORY CLEANUP")
    print("-" * 70)
    root_files = [f for f in root.glob("*") if f.is_file() and f.suffix in [".py", ".pt", ".h5"]]
    
    if root_files:
        print(f"  ⚠ Found {len(root_files)} files still in root:")
        for f in sorted(root_files):
            print(f"    - {f.name}")
    else:
        print("  ✓ Root directory clean (no Python/model files)")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
