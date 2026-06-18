#!/usr/bin/env python3
"""
Project Structure Verification Script
Verifies that all reorganized scripts can find their data/models/outputs
"""

import os
import sys
from pathlib import Path

def check_path(path, description):
    """Check if a path exists"""
    full_path = Path(path)
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def verify_training_scripts():
    """Verify training scripts can find datasets"""
    print("\n=== TRAINING SCRIPTS ===")
    checks = [
        ("../datasets/dataset_cleaned", "Dataset for training"),
        ("../models", "Models directory"),
        ("../outputs", "Output directory"),
        ("train_damage_detection_v1.py", "Main training script"),
    ]
    
    for path, desc in checks:
        check_path(path, desc)

def verify_evaluation_scripts():
    """Verify evaluation scripts can find models"""
    print("\n=== EVALUATION SCRIPTS ===")
    checks = [
        ("../datasets/dataset_cleaned", "Dataset for evaluation"),
        ("../models/best_damage_model.pt", "Model file"),
        ("../outputs", "Output directory"),
        ("evaluate_model.py", "Evaluation script"),
    ]
    
    for path, desc in checks:
        check_path(path, desc)

def verify_preprocessing_scripts():
    """Verify preprocessing scripts can find data"""
    print("\n=== PREPROCESSING SCRIPTS ===")
    checks = [
        ("../datasets", "Datasets directory"),
        ("../models", "Models directory"),
        ("dataset_cleaner.py", "Data cleaner"),
        ("transforms.py", "Transforms"),
    ]
    
    for path, desc in checks:
        check_path(path, desc)

def verify_integration():
    """Verify integration package is intact"""
    print("\n=== INTEGRATION PACKAGE ===")
    integration_path = Path("../integration_package")
    
    required_files = [
        "best_damage_model.pt",
        "inspection_api.py",
        "inspection_pipeline.py",
        "classes.txt",
        "data.yaml",
        "requirements.txt",
        "README.md",
    ]
    
    for fname in required_files:
        fpath = integration_path / fname
        check_path(str(fpath), f"Integration: {fname}")

def verify_structure():
    """Verify complete directory structure"""
    print("\n=== PROJECT STRUCTURE ===")
    base_dirs = [
        "training",
        "evaluation",
        "preprocessing",
        "datasets",
        "models",
        "reports",
        "docs",
        "outputs",
        "archive",
        "integration_package",
    ]
    
    for dir_name in base_dirs:
        check_path(f"../{dir_name}", f"Directory: {dir_name}")

def main():
    """Run all verification checks"""
    print("=" * 60)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 60)
    
    # Check current location
    cwd = Path.cwd()
    print(f"\nCurrent directory: {cwd}")
    print(f"Basename: {cwd.name}")
    
    # Run checks based on current directory
    if "training" in str(cwd):
        print("\n→ Running from training directory")
        verify_training_scripts()
    elif "evaluation" in str(cwd):
        print("\n→ Running from evaluation directory")
        verify_evaluation_scripts()
    elif "preprocessing" in str(cwd):
        print("\n→ Running from preprocessing directory")
        verify_preprocessing_scripts()
    elif "integration_package" in str(cwd):
        print("\n→ Running from integration_package directory")
        verify_integration()
    else:
        print("\n→ Running from project root")
        verify_structure()
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
