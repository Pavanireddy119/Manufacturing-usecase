# Project Audit Report

Generated: 2026-06-18  
Project: Manufacturing Quality Assurance - Vehicle Damage Inspection  
Workspace: `C:\Users\TS6201_TEJASWINI\Desktop\Qaulity_Accurance\Manufacturing-usecase`

## 1. Executive Summary

This project is a vehicle defect inspection system built through several experimental phases:

1. A basic TensorFlow/MobileNetV2 image classifier.
2. An initial YOLOv8 object detector trained on a small cleaned 5-class dataset.
3. A larger YOLOv8 dataset with 18 mixed classes: damage type, vehicle location, and severity.
4. A corrected YOLOv8 dataset that keeps only damage type classes for detection.
5. A serving/reporting layer that enriches YOLO detections with location, severity, recommendations, JSON/PDF reports, annotated image output, a Streamlit dashboard, and a FastAPI endpoint.

The important architectural conclusion is that YOLO should not be trained with damage type, location, and severity as independent classes on the same bounding box. The current corrected direction is:

```text
YOLO detects damage type and bounding box
Post-processing estimates location and severity
Business rules generate recommendation
API/report/dashboard present the inspection result
```

The ML model is not manufacturing-grade yet. The fixed YOLOv8s model exists and can produce detections, but its metrics remain weak based on available results. The pipeline and integration layer are useful prototypes, while production readiness requires stronger data, validated labels, real evaluation targets, and replacement of heuristic location/severity with trained or rules-validated components.

## 2. Project Purpose

The project aims to inspect vehicle images and answer:

- What is damaged?
- Where is it damaged?
- How severe is it?
- What action should be taken?

Target output includes:

```json
{
  "inspection_status": "FAILED",
  "damage_detected": true,
  "damage_type": "paint_damage",
  "damage_location": "front_bumper",
  "severity": "medium",
  "confidence": 0.95,
  "bounding_box": [10, 20, 200, 120],
  "recommendation": "Repaint affected area",
  "timestamp": "..."
}
```

## 3. Architecture Overview

Current intended architecture:

```text
Image Upload / Image Path
        |
        v
YOLOv8 Damage Type Detection
        |
        v
Bounding Box Extraction
        |
        v
Location Classification
        |
        v
Severity Estimation
        |
        v
Business Rules Engine
        |
        v
Inspection API Response
        |
        +--> inspection_report.json
        +--> inspection_report.pdf
        +--> annotated_image.jpg
        +--> Streamlit dashboard
        +--> FastAPI endpoint
```

Observed implementation layers:

- ML training layer: `train_fixed_yolo.py`, `rebuild_dataset.py`, `evaluate_fixed_model.py`, model weights under `output/yolo_fixed/weights/`.
- Historical ML layer: `train_model.py`, `train_damage_detection_v1.py`, `train_yolo.py`, `retrain_yolo.py`, older models and reports.
- Inference/reporting layer: `inspection_pipeline.py`, `dashboard_app.py`, `inspection_api.py`.
- Dataset preparation/audit layer: `create_labeled_dataset.py`, `create_production_dataset.py`, `restructure_dataset.py`, `restructure_for_yolo.py`, `dataset_cleaner.py`, `dataset_audit_full.py`, `analyze_cleaned_dataset.py`.

## 4. File Inventory

Recursive scan summary:

| Item | Count / Size |
| --- | ---: |
| Total files | 19,414 |
| Total size | ~1.42 GB |
| `.JPEG` files | 9,681 |
| `.jpg` files | 4,953 |
| `.txt` files | 4,685 |
| Python scripts | 34 |
| PNG artifacts | 11 |
| YAML configs | 10 |
| PyTorch YOLO model files | 10 |
| CSV files | 9 |
| Cache files | 6 |
| JSON files | 4 |
| Python bytecode files | 3 |
| ZIP archives | 1 |
| Keras H5 model files | 1 |
| PDF files | 1 |

Top-level directory inventory:

| Path | File Count | Approx Size | Category |
| --- | ---: | ---: | --- |
| `Dataset/` | 2,300 | 130 MB | Original/source dataset |
| `preprocessed_dataset/` | 2,300 | 24 MB | Preprocessed image dataset |
| `train/` | 1,840 | 19 MB | Keras train split |
| `test/` | 460 | 5 MB | Keras test split |
| `dataset_labeled/` | 5,336 | 259 MB | Auto/manual labeled intermediate dataset |
| `dataset_final/` | 2,011 | 12 MB | 18-class mixed YOLO dataset |
| `dataset_yolo_restructured/` | 1,789 | 8 MB | Intermediate 6-class damage-only YOLO dataset |
| `dataset_yolo_fixed/` | 1,905 | 9 MB | Current fixed YOLO dataset |
| `dataset_cleaned/` | 94 | <1 MB | Older tiny 5-class YOLO dataset |
| `new_dataset/` | 1,265 | 6 MB | Intermediate generated dataset |
| `output/` | 43 | 343 MB | Models, reports, predictions, inspection artifacts |
| `runs/` | 19 | 15 MB | Ultralytics training run output |
| `test_output/` | 1 | <1 MB | Test prediction image |
| `__pycache__/` | 3 | small | Generated Python bytecode |

Dataset split inventory:

| Dataset | Train Images | Train Labels | Val Images | Val Labels | Test Images | Test Labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dataset_cleaned/` | 36 | 36 | 7 | 7 | 2 | 2 |
| `dataset_yolo_fixed/` | 721 | 721 | 173 | 173 | 57 | 57 |
| `dataset_yolo_restructured/` | 721 | 721 | 173 | 173 | 0 | 0 |
| `dataset_final/` | 721 | 721 | 173 | 173 | 57 | 57 |
| `dataset_labeled/` | 1,712 | 920 | 429 | 230 | 85 | 43 |
| `new_dataset/` | 475 | 475 | 117 | 117 | 38 | 38 |

Model inventory:

| Model File | Size | Category | Notes |
| --- | ---: | --- | --- |
| `output/yolo_fixed/weights/best.pt` | ~67 MB | Current candidate YOLO model | Fixed 6-class YOLOv8s path. |
| `output/yolo_fixed/weights/last.pt` | ~67 MB | Training checkpoint | Same run as fixed model. |
| `output/models/weights/best.pt` | ~67 MB | Older YOLO model | Mixed 18-class experiment. |
| `output/models/weights/last.pt` | ~67 MB | Older YOLO checkpoint | Mixed 18-class experiment. |
| `output/models/best.pt` | ~67 MB | Older YOLO copy | Duplicate/legacy output. |
| `runs/detect/damage_detection_v1/weights/best.pt` | ~6 MB | Failed YOLOv8n model | Tiny 5-class dataset attempt. |
| `runs/detect/damage_detection_v1/weights/last.pt` | ~6 MB | Failed YOLOv8n checkpoint | Same failed run. |
| `best_damage_model.pt` | ~6 MB | Failed YOLOv8n copy | Duplicate of `runs/.../best.pt`. |
| `vehicle_damage_model.h5` | ~11 MB | Keras classifier | Binary MobileNetV2 classifier, not object detector. |
| `yolov8n.pt` | ~6 MB | Base pretrained weight | Useful for training/reference. |
| `yolov8s.pt` | ~23 MB | Base pretrained weight | Used by YOLOv8s training. |

Python script inventory:

| File | Category | Purpose |
| --- | --- | --- |
| `inspection_pipeline.py` | Production candidate | End-to-end inspection, enrichment, business rules, reports, annotated image. |
| `dashboard_app.py` | Production candidate | Streamlit upload/dashboard UI. |
| `inspection_api.py` | Production candidate | FastAPI model-serving endpoint. |
| `train_fixed_yolo.py` | Production candidate | Current YOLOv8s training script for fixed 6-class dataset. |
| `evaluate_fixed_model.py` | Production candidate | Evaluation and sample prediction script for fixed YOLO model. |
| `rebuild_dataset.py` | Production candidate | Rebuilds `dataset_yolo_fixed/` from `dataset_final/`, removing triplicate mixed annotations. |
| `predict_damage.py` | Supporting/legacy | Earlier inference/report script with 18-class grouping assumptions. |
| `train_yolo.py` | Obsolete/legacy | Trains on mixed 18-class dataset; should not be used for final training. |
| `restructure_for_yolo.py` | Supporting/obsolete | Earlier damage-only dataset restructuring without test split. Superseded by `rebuild_dataset.py`. |
| `retrain_yolo.py` | Experimental | Retraining wrapper around restructured dataset. |
| `train_damage_detection_v1.py` | Obsolete | Failed YOLOv8n training on tiny `dataset_cleaned/`. |
| `train_model.py` | Obsolete | MobileNetV2 binary classifier; not aligned with current object-detection requirement. |
| `build_model.py` | Obsolete/supporting | Keras model construction helper. |
| `evaluate_model.py` | Obsolete/supporting | Keras classifier evaluation. |
| `evaluate_yolo.py` | Supporting/legacy | YOLO evaluation for mixed dataset path. |
| `evaluate_model_comprehensive.py` | Supporting/legacy | Root-cause evaluation of failed YOLOv8n run. |
| `test_prediction.py` | Supporting | Quick one-image prediction smoke test. |
| `generate_report.py` | Supporting | Report generation and plots for earlier dataset/training analysis. |
| `generate_metrics.py` | Supporting | Metric generation helper. |
| `diagnose_yolo.py` | Supporting | Diagnostic script for bad YOLO behavior. |
| `dataset_audit_full.py` | Supporting | Dataset audit utility. |
| `dataset_cleaner.py` | Obsolete/supporting | Older cleanup path into `dataset_cleaned/`. |
| `analyze_cleaned_dataset.py` | Supporting | Statistics for cleaned dataset. |
| `create_labeled_dataset.py` | Experimental/supporting | Large labeling/data generation utility. |
| `create_production_dataset.py` | Experimental/supporting | Larger generated dataset builder. |
| `restructure_dataset.py` | Experimental/supporting | Dataset restructuring utility. |
| `dataset_analysis.py` | Supporting | Simple dataset analysis utility. |
| `image_size_analysis.py` | Supporting | Image dimension analysis. |
| `resize_images.py` | Supporting | Image resizing utility. |
| `create_train_test.py` | Supporting/obsolete | Earlier train/test split utility. |
| `check_corrupted.py` | Supporting | Corruption check utility. |
| `verify_load_dataset.py` | Supporting | Dataset loading check. |
| `verify_output.py` | Supporting | Output verification utility. |
| `transforms.py` | Supporting | Small transform helper. |

Report and documentation inventory:

| File | Category | Purpose |
| --- | --- | --- |
| `COMPREHENSIVE_EVALUATION_REPORT.txt` | Supporting | Detailed report for earlier failed 5-class YOLOv8n attempt. |
| `DELIVERABLES_MANIFEST.txt` | Supporting | Manifest of generated files and prior status. |
| `EXECUTIVE_SUMMARY.txt` | Supporting | Executive summary of prior state. |
| `PROJECT_COMPLETE.txt` | Supporting | Completion summary from prior iteration. |
| `QUICK_REFERENCE.txt` | Supporting | Quick overview and commands. |
| `root_cause_analysis_report.txt` | Supporting | Root cause analysis for failed small-data run. |
| `prediction_diagnostic_report.txt` | Supporting | Prediction diagnostics. |
| `dataset_audit_report.txt` | Supporting | Dataset audit summary. |
| `class_distribution_cleaned.txt` | Supporting | Class distribution for tiny cleaned dataset. |
| `output/reports/final_diagnosis_report.txt` | Supporting | Diagnosis for mixed 18-class YOLO problem. |
| `output/inspection_sample/inspection_report.json` | Generated output | Sample inspection API/report artifact. |
| `output/inspection_sample/inspection_report.pdf` | Generated output | Sample PDF report. |

Important config files:

| File | Category | Purpose |
| --- | --- | --- |
| `dataset_yolo_fixed/data.yaml` | Production candidate | Current fixed 6-class YOLO dataset config. |
| `dataset_final/data.yaml` | Supporting/obsolete | 18-class mixed dataset config; source for rebuild but not final training. |
| `dataset_final/data_yolov8.yaml` | Supporting/obsolete | Alternate mixed dataset config. |
| `dataset_yolo_restructured/damage_only.yaml` | Supporting | Intermediate 6-class damage-only config. |
| `dataset_cleaned/data.yaml` | Obsolete | Tiny 5-class dataset config. |
| `output/yolo_fixed/args.yaml` | Generated output | Training args for current fixed model. |
| `output/models/args.yaml` | Generated output | Training args for old mixed model. |

Temporary/generated files:

- `__pycache__/`
- `*.pyc`
- `*.cache`
- `output/ultralytics_config/`
- TensorBoard event files under `output/` and `runs/`
- `test_output/`
- `output/predictions/`
- `output/inspection_sample/`

## 5. Production Files

Production candidate files for the final solution:

| File / Directory | Why It Matters |
| --- | --- |
| `inspection_pipeline.py` | Central end-to-end inspection pipeline. |
| `dashboard_app.py` | Upload workflow and inspection dashboard. |
| `inspection_api.py` | API serving layer. |
| `rebuild_dataset.py` | Corrects mixed-label dataset into damage-type-only YOLO dataset. |
| `train_fixed_yolo.py` | Current training entry point. |
| `evaluate_fixed_model.py` | Current evaluation/sample prediction entry point. |
| `dataset_yolo_fixed/` | Current fixed YOLO dataset with train/val/test splits. |
| `dataset_yolo_fixed/data.yaml` | Current YOLO dataset config. |
| `output/yolo_fixed/weights/best.pt` | Current candidate trained YOLOv8s model. |
| `output/yolo_fixed/results.csv` | Current fixed-model training metrics. |
| `yolov8s.pt` | Base pretrained model used for current training. |

Production deliverable artifacts:

| Artifact | Purpose |
| --- | --- |
| `inspection_report.json` | Machine-readable inspection result. |
| `inspection_report.pdf` | Business-readable inspection report. |
| `annotated_image.jpg` | Visual evidence with bbox/type/location/severity/confidence. |

## 6. Supporting Files

Supporting files are useful for traceability but are not part of the core runtime:

- `COMPREHENSIVE_EVALUATION_REPORT.txt`
- `DELIVERABLES_MANIFEST.txt`
- `EXECUTIVE_SUMMARY.txt`
- `PROJECT_COMPLETE.txt`
- `QUICK_REFERENCE.txt`
- `root_cause_analysis_report.txt`
- `prediction_diagnostic_report.txt`
- `dataset_audit_report.txt`
- `class_distribution_cleaned.txt`
- `output/reports/*`
- `generate_report.py`
- `generate_metrics.py`
- `diagnose_yolo.py`
- `dataset_audit_full.py`
- `analyze_cleaned_dataset.py`
- `test_prediction.py`
- `verify_output.py`
- `verify_load_dataset.py`
- `check_corrupted.py`
- `image_size_analysis.py`

## 7. Experimental Files

These files appear to have been created during exploration, dataset generation, or earlier attempts:

- `create_labeled_dataset.py`
- `create_production_dataset.py`
- `restructure_dataset.py`
- `retrain_yolo.py`
- `dataset_labeled/`
- `new_dataset/`
- `preprocessed_dataset/`
- `Dataset.zip`
- `test_output/`
- `output/predictions/`

## 8. Obsolete Files

These are superseded by the current fixed-YOLO pipeline:

- `train_model.py`
- `build_model.py`
- `evaluate_model.py`
- `vehicle_damage_model.h5`
- `train_damage_detection_v1.py`
- `best_damage_model.pt`
- `runs/detect/damage_detection_v1/`
- `dataset_cleaned/`
- `dataset_cleaner.py`
- `train_yolo.py`
- `output/models/`
- `dataset_final/data.yaml` for direct training
- `dataset_final/data_yolov8.yaml` for direct training

Important nuance: `dataset_final/` should not be deleted immediately because it is the source dataset used by `rebuild_dataset.py` to create `dataset_yolo_fixed/`. It is obsolete as a direct training dataset, but still useful as lineage/source material.

## 9. Duplicate Files

Functional duplicates:

| Duplicate Group | Files | Recommendation |
| --- | --- | --- |
| YOLO training scripts | `train_yolo.py`, `retrain_yolo.py`, `train_damage_detection_v1.py`, `train_fixed_yolo.py` | Keep `train_fixed_yolo.py`; archive the rest. |
| Dataset restructuring | `dataset_cleaner.py`, `restructure_for_yolo.py`, `rebuild_dataset.py`, parts of `restructure_dataset.py` | Keep `rebuild_dataset.py`; archive older scripts. |
| Evaluation scripts | `evaluate_model.py`, `evaluate_yolo.py`, `evaluate_model_comprehensive.py`, `evaluate_fixed_model.py` | Keep `evaluate_fixed_model.py`; keep older ones only for traceability. |
| Inference scripts | `predict_damage.py`, `inspection_pipeline.py`, `test_prediction.py` | Keep `inspection_pipeline.py`; keep `test_prediction.py` as smoke test; archive `predict_damage.py` or migrate useful diagnostics. |
| Model copies | `best_damage_model.pt`, `runs/.../best.pt`; several `output/models/*.pt` | Keep only current fixed model plus pretrained bases. |
| Dataset variants | `dataset_cleaned/`, `dataset_yolo_restructured/`, `dataset_yolo_fixed/`, `dataset_final/` | Keep `dataset_yolo_fixed/` and source lineage; archive older derived datasets. |

## 10. Workflow Diagram

Real workflow discovered from code and reports:

```text
Original Dataset / Dataset.zip
        |
        v
Dataset exploration
  - dataset_analysis.py
  - image_size_analysis.py
  - check_corrupted.py
        |
        v
Keras classification path
  - create_train_test.py
  - train_model.py
  - vehicle_damage_model.h5
  Status: obsolete; binary classifier only
        |
        v
YOLO dataset generation
  - create_labeled_dataset.py
  - create_production_dataset.py
  - dataset_labeled/
  - dataset_final/
        |
        v
Initial YOLO training on mixed 18-class labels
  - train_yolo.py
  - output/models/
  Result: poor performance because damage, location, severity were separate YOLO classes
        |
        v
Diagnosis
  - diagnose_yolo.py
  - output/reports/final_diagnosis_report.txt
  Finding: triplicate annotations and mixed class semantics break YOLO learning
        |
        v
Dataset rebuild
  - rebuild_dataset.py
  - dataset_yolo_fixed/
  Fix: keep only damage type classes
        |
        v
Fixed YOLO training
  - train_fixed_yolo.py
  - output/yolo_fixed/weights/best.pt
        |
        v
Fixed model evaluation
  - evaluate_fixed_model.py
  - output/yolo_fixed/results.csv
        |
        v
Inspection integration
  - inspection_pipeline.py
  - dashboard_app.py
  - inspection_api.py
        |
        v
Business deliverables
  - inspection_report.json
  - inspection_report.pdf
  - annotated_image.jpg
```

## 11. Deliverables

Current technical deliverables:

| Deliverable | Status | Purpose |
| --- | --- | --- |
| `vehicle_damage_model.h5` | Historical/obsolete | Binary Keras vehicle damage classifier. Does not localize defects. |
| `best_damage_model.pt` | Historical/failed | YOLOv8n trained on tiny 5-class dataset. Reported non-functional. |
| `output/models/weights/best.pt` | Historical/weak | YOLOv8s trained on mixed 18-class dataset. Not suitable for final use. |
| `output/yolo_fixed/weights/best.pt` | Current candidate | YOLOv8s trained on fixed 6-class damage dataset. |
| `dataset_yolo_fixed/` | Current dataset | Damage-type-only YOLO dataset with train/val/test splits. |
| `inspection_report.json` | Current integration output | Structured API-style result. |
| `inspection_report.pdf` | Current integration output | Human-readable inspection report. |
| `annotated_image.jpg` | Current integration output | Visual result with bounding boxes and labels. |
| `dashboard_app.py` | Current integration output | Streamlit dashboard. |
| `inspection_api.py` | Current integration output | FastAPI service. |

## 12. Current Status

Completed:

- Dataset lineage and diagnosis are documented.
- Root cause of mixed-label YOLO failure is identified.
- Corrected damage-type-only YOLO dataset exists as `dataset_yolo_fixed/`.
- Current YOLOv8s fixed-model training output exists at `output/yolo_fixed/weights/best.pt`.
- End-to-end inspection pipeline exists.
- JSON/PDF/image report generation exists.
- Streamlit dashboard exists.
- FastAPI serving wrapper exists.

Partially completed:

- Damage type detection: implemented, but model quality is not production-grade.
- Location classification: implemented as heuristic geometry, not trained classifier.
- Severity classification: implemented as heuristic area/damage criticality, not trained classifier.
- Business recommendations: implemented as rule-based logic; needs stakeholder validation.
- Evaluation: scripts exist, but production acceptance thresholds and robust test-set metrics are not established.

Pending:

- Collect or curate a larger validated dataset, especially for `scratch`, `paint_damage`, and `broken_part`.
- Establish a frozen test set with real manufacturing acceptance criteria.
- Improve model metrics and calibrate confidence thresholds.
- Validate location/severity logic against labeled ground truth or train separate classifiers/segmenters.
- Add automated regression tests for inference schema and report generation.
- Add dependency management such as `requirements.txt` or environment file.
- Add deployment documentation.
- Remove or archive obsolete artifacts.

Should not be changed casually:

- `dataset_yolo_fixed/` until a newer validated dataset replaces it.
- `output/yolo_fixed/weights/best.pt` until a better model is trained and benchmarked.
- `rebuild_dataset.py` because it captures the critical data correction.
- `inspection_pipeline.py` public response schema unless API versioning is introduced.

## 13. Team Responsibilities

### ML Team

Responsible for:

- Dataset acquisition and labeling strategy.
- Dataset audit and class balance.
- `dataset_yolo_fixed/` and future dataset versions.
- Training scripts: `rebuild_dataset.py`, `train_fixed_yolo.py`.
- Evaluation scripts: `evaluate_fixed_model.py`.
- Model artifacts: `output/yolo_fixed/weights/best.pt`, future promoted checkpoints.
- Metrics, threshold selection, model cards, and acceptance criteria.

ML team should own:

- Damage type taxonomy.
- Annotation guidelines.
- Train/val/test split policy.
- Model quality gates.
- Confidence calibration.

### Backend Team

Responsible for:

- `inspection_api.py`.
- Model loading and inference serving.
- Request/response schema.
- Artifact storage and retrieval.
- Runtime configuration.
- Logging, monitoring, latency, and error handling.
- API versioning and security.

Backend team should own:

- `/inspect` endpoint.
- Health checks.
- Deployment container or service.
- Report artifact paths.
- Production model selection/configuration.

### Frontend Team

Responsible for:

- `dashboard_app.py`.
- Upload workflow.
- Inspection result presentation.
- Download links for JSON/PDF/image.
- User-facing status and failure handling.

Frontend team should own:

- Dashboard layout and usability.
- Visual hierarchy for damage type, location, severity, confidence, recommendation.
- Operator workflow for pass/fail inspection review.

### Business / Quality Team

Responsible for:

- Repair recommendation rules.
- Severity definitions.
- Acceptance/rejection thresholds.
- Human review policy.
- Manufacturing disposition workflow.

## 14. Integration Readiness

Integration prototype readiness: moderate.

The integration layer can demonstrate the desired end-to-end flow:

- Upload image.
- Run YOLO detection.
- Estimate location.
- Estimate severity.
- Generate recommendation.
- Produce JSON/PDF/JPG artifacts.
- Display result in a dashboard.
- Serve result through FastAPI.

Production readiness: not yet.

Reasons:

- Model metrics are weak.
- Location and severity are heuristic.
- No robust automated tests.
- No dependency file.
- No documented deployment package.
- No model registry or versioning.
- No formal validation set with sign-off.

## 15. Risks

High risks:

- Model may miss defects or generate false positives due to weak training data and metrics.
- `scratch` has historically been missing or severely underrepresented.
- Location/severity are not learned from validated labels.
- Multiple old models can cause accidental loading of the wrong checkpoint.
- Multiple dataset variants can confuse future training.
- Existing reports contain conflicting status narratives because they refer to different experiment phases.

Medium risks:

- Absolute Windows paths in dataset YAML files reduce portability.
- No dependency manifest makes environment recreation fragile.
- Generated artifacts and source files are mixed in the repository root.
- Lack of tests can break API/report schema silently.

Low risks:

- TensorBoard logs and cache files are harmless but clutter the workspace.
- Pretrained base weights are useful but should be documented.

## 16. Recommendations

Immediate recommendations:

1. Declare `dataset_yolo_fixed/`, `train_fixed_yolo.py`, `evaluate_fixed_model.py`, and `output/yolo_fixed/weights/best.pt` as the current ML baseline.
2. Declare `inspection_pipeline.py`, `inspection_api.py`, and `dashboard_app.py` as the current integration baseline.
3. Archive obsolete training attempts into an `archive/` directory rather than deleting immediately.
4. Add `requirements.txt`.
5. Add a simple smoke test for:
   - Model loads.
   - API response contains required keys.
   - JSON/PDF/JPG artifacts are generated.
6. Create a model selection config so old checkpoints are not loaded accidentally.

ML recommendations:

1. Build a larger manually verified dataset.
2. Ensure all required classes have enough examples:
   - `dent`
   - `scratch`
   - `crack`
   - `broken_part`
   - `paint_damage`
3. Keep damage type detection as YOLO classes.
4. Treat location and severity as separate tasks unless reliable labels exist.
5. Define acceptance targets before retraining:
   - mAP50.
   - precision.
   - recall.
   - per-class AP.
   - false reject / false accept rates.

Architecture recommendations:

1. Split the project into clear folders:

```text
src/
  inspection/
  api/
  dashboard/
scripts/
  data/
  training/
  evaluation/
models/
data/
reports/
archive/
```

2. Keep generated training outputs out of the root.
3. Add model metadata:

```json
{
  "model_name": "vehicle_damage_yolov8s",
  "dataset": "dataset_yolo_fixed",
  "classes": ["dent", "scratch", "crack", "broken_part", "paint_damage", "other_damage"],
  "trained_at": "...",
  "metrics": {}
}
```

## 17. Cleanup Plan

Phase 1: Archive only, no deletion.

Move to `archive/old_keras/`:

- `train_model.py`
- `build_model.py`
- `evaluate_model.py`
- `vehicle_damage_model.h5`
- `train/`
- `test/`

Move to `archive/failed_yolov8n_small_dataset/`:

- `train_damage_detection_v1.py`
- `best_damage_model.pt`
- `runs/detect/damage_detection_v1/`
- `dataset_cleaned/`
- `dataset_cleaner.py`
- `COMPREHENSIVE_EVALUATION_REPORT.txt`
- `root_cause_analysis_report.txt`
- `class_distribution_cleaned.txt`

Move to `archive/mixed_18_class_yolo/`:

- `train_yolo.py`
- `output/models/`
- old mixed-class training reports

Move to `archive/dataset_experiments/`:

- `dataset_yolo_restructured/`
- `new_dataset/`
- `dataset_labeled/`
- `preprocessed_dataset/`
- `create_labeled_dataset.py`
- `create_production_dataset.py`
- `restructure_dataset.py`
- `restructure_for_yolo.py`
- `retrain_yolo.py`

Keep active:

- `dataset_final/` as source lineage for rebuild.
- `dataset_yolo_fixed/`.
- `rebuild_dataset.py`.
- `train_fixed_yolo.py`.
- `evaluate_fixed_model.py`.
- `inspection_pipeline.py`.
- `inspection_api.py`.
- `dashboard_app.py`.
- `output/yolo_fixed/weights/best.pt`.
- `output/yolo_fixed/results.csv`.
- `yolov8s.pt`.

Phase 2: Add reproducibility.

- Add `requirements.txt`.
- Add `README.md`.
- Add `configs/inspection.yaml`.
- Add tests for inference/report generation.
- Add `models/current/` or explicit model path config.

Phase 3: Production hardening.

- Replace heuristic location/severity with validated classifiers or business-approved rules.
- Add model registry/versioning.
- Add monitoring for confidence, class distribution, and failed inspections.
- Add human-review workflow for low-confidence predictions.

## 18. Final Assessment

The project has a viable architecture and a working prototype pipeline, but the ML task is not truly complete. The core design correction has been made: detect damage type with YOLO and enrich location/severity afterward. However, manufacturing-grade readiness requires stronger labeled data, better per-class metrics, validated severity/location outputs, and a cleaner deployable project structure.

The next logical step is not refactoring code first. The next step should be to freeze the current baseline, define production acceptance metrics, and validate or improve the dataset/model. After that, cleanup and refactoring can be done safely around a known-good baseline.
