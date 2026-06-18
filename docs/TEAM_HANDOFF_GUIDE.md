# Team Handoff Guide

Generated: 2026-06-18

## Backend Team

Required deployment files are in `integration_package/`:

- `inspection_pipeline.py` - core inspection pipeline.
- `inspection_api.py` - FastAPI upload/inference API.
- `dashboard_app.py` - Streamlit inspection dashboard.
- `best_damage_model.pt` - packaged YOLO model.
- `classes.txt` - damage class labels.
- `data.yaml` - portable dataset metadata.
- `transforms.py` - preprocessing transform hook.
- `requirements.txt` - integration dependencies.
- `README.md` - package usage notes.

Model search order now checks the package model first and then `models/best_damage_model.pt`.

## ML Team

Training scripts are in `training/`:

- `train_damage_detection_v1.py`
- `train_fixed_yolo.py`
- `train_model.py`
- `train_yolo.py`
- `retrain_yolo.py`
- `build_model.py`

Evaluation scripts are in `evaluation/`:

- `evaluate_model.py`
- `evaluate_fixed_model.py`
- `evaluate_yolo.py`
- `evaluate_model_comprehensive.py`
- `generate_metrics.py`
- `verify_output.py`

Preprocessing scripts are in `preprocessing/`:

- `dataset_cleaner.py`
- `rebuild_dataset.py`
- `resize_images.py`
- `create_train_test.py`
- `create_production_dataset.py`
- `create_labeled_dataset.py`
- `restructure_dataset.py`
- `restructure_for_yolo.py`
- `transforms.py`

Datasets are centralized in `datasets/`, including `train/`, `test/`, `validation/`, `dataset_cleaned/`, `dataset_yolo_fixed/`, `dataset_labeled/`, `dataset_yolo_restructured/`, and `preprocessed_dataset/`.

Model artifacts are centralized in `models/`:

- `best_damage_model.pt`
- `vehicle_damage_model.h5`
- `yolov8n.pt`
- `yolov8s.pt`

## Frontend Team

UI/dashboard entrypoint:

- `integration_package/dashboard_app.py`

Dashboard-generated artifacts should be written to `outputs/inspection_reports/`.

## Management

Primary reports:

- `reports/EXECUTIVE_SUMMARY.txt`
- `reports/QUICK_REFERENCE.txt`
- `reports/COMPREHENSIVE_EVALUATION_REPORT.txt`
- `reports/PROJECT_COMPLETE.txt`
- `reports/root_cause_analysis_report.txt`
- `reports/dataset_audit_report.txt`
- `reports/PROJECT_AUDIT_REPORT.md`

Primary docs:

- `docs/PROJECT_STRUCTURE_REPORT.md`
- `docs/TEAM_HANDOFF_GUIDE.md`
- `docs/CLEANUP_RECOMMENDATIONS.md`
- `docs/FINAL_VALIDATION_REPORT.md`
- `docs/QUICK_START_INTEGRATION.md`
- `docs/INTEGRATION_PACKAGE_REPORT.md`
- `docs/INTEGRATION_DEPLOYMENT_SUMMARY.md`

## Preservation Policy

Do not delete archived files until the team has reviewed them. Legacy and uncertain files are preserved in `archive/`.
