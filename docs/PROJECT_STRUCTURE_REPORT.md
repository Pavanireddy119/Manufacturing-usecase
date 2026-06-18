# Project Structure Report

Generated: 2026-06-18

## Summary

The repository has been reorganized into a production-oriented layout for ML development, backend integration, reporting, and future retraining. No files were deleted. Root-level duplicate scripts, datasets, model artifacts, and legacy outputs were moved into `archive/` buckets for preservation.

## Old Structure

Before final cleanup, the project root contained mixed concerns:

- Training, evaluation, preprocessing, API, and dashboard scripts at the root.
- Multiple dataset folders at the root, including `Dataset/`, `dataset_cleaned/`, `dataset_final/`, `dataset_labeled/`, `dataset_yolo_fixed/`, `dataset_yolo_restructured/`, `preprocessed_dataset/`, `train/`, and `test/`.
- Model files at the root, including `vehicle_damage_model.h5`, `yolov8n.pt`, and `yolov8s.pt`.
- Reports and integration docs duplicated between root-level files and organized folders.
- Legacy outputs under root-level `output/`, `runs/`, and `test_output/`.

## New Structure

The root now contains only the main ownership folders and root project files:

- `integration_package/` - backend/API inspection package and deployable model bundle.
- `datasets/` - active and preserved training/evaluation datasets.
- `models/` - central model artifacts.
- `training/` - model training and retraining scripts.
- `evaluation/` - evaluation, metrics, and verification scripts.
- `preprocessing/` - dataset preparation and transformation scripts.
- `reports/` - management and audit reports.
- `docs/` - handoff, integration, and structure documentation.
- `outputs/` - predictions, inspection artifacts, annotated images, and model outputs.
- `archive/` - preserved duplicate, legacy, or uncertain files.
- `README.md`, `.gitignore`, and `requirements.txt`.

## Files Moved

- Root training scripts moved or preserved under `training/`; root duplicates archived under `archive/deprecated_scripts/root_duplicates/`.
- Root evaluation scripts moved or preserved under `evaluation/`; root duplicates archived under `archive/deprecated_scripts/root_duplicates/`.
- Root preprocessing scripts moved or preserved under `preprocessing/`; root duplicates archived under `archive/deprecated_scripts/root_duplicates/`.
- Root integration scripts moved or preserved under `integration_package/`; root duplicates archived under `archive/deprecated_scripts/root_duplicates/`.
- Root reports moved to `reports/` or archived under `archive/old_experiments/root_report_duplicates/` when organized copies already existed.
- Root integration docs moved to `docs/` or archived under `archive/old_experiments/root_doc_duplicates/` when organized copies already existed.
- Root model artifacts moved to `archive/legacy_models/root_duplicates/` after confirming curated copies exist in `models/`.
- Root dataset/output folders moved to `archive/duplicate_datasets/root_duplicates/` after confirming curated dataset/output folders exist.
- Legacy `outputs/output/` contents were normalized into `outputs/predictions/`, `outputs/inspection_reports/`, and `outputs/model_outputs/`.

## Reference Updates

Script references were updated from old root paths to the new layout:

- Dataset paths now use `datasets/...`.
- Model paths now use `models/...` or packaged `integration_package/best_damage_model.pt`.
- Training and run artifacts now use `outputs/model_outputs/...`.
- Prediction artifacts now use `outputs/predictions/...`.
- Inspection reports now use `outputs/inspection_reports/...`.
- Absolute local paths were replaced with portable relative paths.

## Rationale

This structure separates team responsibilities cleanly:

- ML engineers can work in `training/`, `evaluation/`, `preprocessing/`, `datasets/`, and `models/`.
- Backend engineers can deploy from `integration_package/`.
- Frontend/dashboard work is discoverable in `integration_package/dashboard_app.py`.
- Managers can read reports in `reports/` and handoff docs in `docs/`.
- Future retraining can reuse canonical datasets and scripts without relying on root-level duplicates.
