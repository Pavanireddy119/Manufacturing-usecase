# Manufacturing Visual Inspection

Production-ready repository for vehicle damage detection, inspection API integration, model retraining, evaluation, and project reporting.

## Quick Start

Backend/API:

```powershell
cd integration_package
pip install -r requirements.txt
uvicorn inspection_api:app --port 8000
```

ML training and evaluation:

```powershell
python training/train_damage_detection_v1.py
python evaluation/evaluate_yolo.py
```

The full handoff guide is in `docs/TEAM_HANDOFF_GUIDE.md`.

## Project Structure

```text
Manufacturing-usecase/
  integration_package/   Backend/API inspection package
  datasets/              Training, test, validation, and prepared datasets
  models/                Central model artifacts
  training/              Training and retraining scripts
  evaluation/            Evaluation and metric scripts
  preprocessing/         Dataset preparation scripts
  reports/               Audit, executive, and evaluation reports
  docs/                  Integration and team handoff documentation
  outputs/               Predictions, reports, annotations, model outputs
  archive/               Preserved legacy, duplicate, and uncertain files
  README.md
  .gitignore
  requirements.txt
```

## Important Files

- Integration package: `integration_package/README.md`
- Team handoff: `docs/TEAM_HANDOFF_GUIDE.md`
- Structure report: `docs/PROJECT_STRUCTURE_REPORT.md`
- Cleanup recommendations: `docs/CLEANUP_RECOMMENDATIONS.md`
- Validation report: `docs/FINAL_VALIDATION_REPORT.md`
- Executive summary: `reports/EXECUTIVE_SUMMARY.txt`

## Models

Central models are stored in `models/`:

- `best_damage_model.pt`
- `vehicle_damage_model.h5`
- `yolov8n.pt`
- `yolov8s.pt`

The deployable integration copy is stored at `integration_package/best_damage_model.pt`.

## Preservation Policy

No files were deleted during restructuring. Duplicate or uncertain files were moved into `archive/` for review.
