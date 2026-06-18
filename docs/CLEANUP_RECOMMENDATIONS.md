# Cleanup Recommendations

Generated: 2026-06-18

## Safe To Keep

- `integration_package/` for backend deployment.
- `datasets/` for active and historical dataset variants.
- `models/` for central model artifacts.
- `training/`, `evaluation/`, and `preprocessing/` for ML workflows.
- `reports/` and `docs/` for management, audit, and handoff materials.
- `outputs/` for generated predictions, inspection artifacts, annotated images, and model outputs.

## Safe To Archive

Already archived for review:

- Root duplicate scripts: `archive/deprecated_scripts/root_duplicates/`.
- Maintenance verification scripts: `archive/deprecated_scripts/maintenance/`.
- Root duplicate datasets and output folders: `archive/duplicate_datasets/root_duplicates/`.
- Root duplicate model files: `archive/legacy_models/root_duplicates/`.
- Superseded or duplicate reports/docs: `archive/old_experiments/`.
- Legacy output wrapper folders: `archive/old_experiments/legacy_output_wrappers/`.

## Do Not Delete

Do not delete these without explicit team approval:

- Any folder under `datasets/`.
- Any model under `models/` or `integration_package/best_damage_model.pt`.
- Archived duplicate datasets until dataset lineage is confirmed.
- Archived legacy models until model registry/versioning is finalized.
- Existing reports and audit outputs.

## Recommended Next Steps

- Add lightweight smoke-test commands to CI once dependencies are standardized.
- Decide whether `dataset_*_old` folders under `datasets/` should remain active lineage or move into `archive/duplicate_datasets/`.
- Consider a model registry file that records model name, path, training dataset, date, and metrics.
- Add a short `datasets/README.md` describing which dataset variant is canonical for retraining.
