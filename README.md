# QualityVision AI — Manufacturing Visual Inspection

AI-powered visual quality inspection platform. A user uploads a vehicle image,
the system runs a YOLOv8 damage-detection model, and returns a quality verdict
(`Defective` / `Non-Defective`) with damage type, location, severity and a
repair recommendation. Includes authentication, image upload and per-user
prediction history.

This `integration` branch unifies the three feature branches of the project:

| Module | Path | Stack | Source branch |
| --- | --- | --- | --- |
| **Backend API** | [`backend/`](backend/) | FastAPI · SQLAlchemy · JWT | `backend` |
| **Frontend** | [`frontend/`](frontend/) | React 19 · Vite · React Router | `frontend` |
| **ML model & data** | [`ml/`](ml/) | YOLOv8 (Ultralytics/PyTorch) | `data-preprocessing` |

## Architecture

```
Browser (React/Vite)
        │  REST + JWT
        ▼
FastAPI backend (backend/)
  ├─ /api/auth         signup / login
  ├─ /api/upload       store image, return image_id
  ├─ /api/predict/{id} run inspection on a stored image
  └─ /api/history      past predictions
        │  in-process call
        ▼
YoloDamagePredictor (backend/app/ml/yolo_predictor.py)
        │  imports
        ▼
ml/integration_package/inspection_pipeline.py
  YOLOv8 detection → location → severity → business rules → result
        │
        ▼
ml/integration_package/best_damage_model.pt
```

The backend calls the ML pipeline **in-process**. If `torch` / `ultralytics`
or the model weights are unavailable, the predictor transparently falls back to
a placeholder so the API never fails to start or respond.

## Quick start

### 1. Backend + ML (Python 3.10 or 3.11 — not 3.14, no torch wheels yet)

```bash
cd backend
py -3.11 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt            # core API
pip install -r requirements-ml.txt         # real YOLOv8 model (torch, ultralytics, ...)
copy .env.example .env
uvicorn app.main:app --reload
```

- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

> Skip `requirements-ml.txt` to run the API with the lightweight placeholder
> predictor (`ML_ENABLED=false` also forces this).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

## API flow

1. `POST /api/auth/signup` then `POST /api/auth/login` → access token
2. `POST /api/upload` (Bearer token) → `image_id`
3. `POST /api/predict/{image_id}` → quality verdict + damage detail
4. `GET /api/history` → previous predictions

## ML model

The active model is a **YOLOv8 binary image classifier**
(`ml/integration_package/quality_classifier.pt`, ~3 MB) trained on the balanced
`Dataset/data1a` set (1,840 train / 460 val images, damage vs whole). It reaches
**~95% validation accuracy** and correctly separates defective from
non-defective vehicles with high confidence.

- Retrain / reproduce: `cd ml && python train_quality_classifier.py`
  (rebuilds the train/val layout from `Dataset/data1a` and trains via transfer
  learning from `yolov8n-cls.pt`).
- The predictor ([`backend/app/ml/yolo_predictor.py`](backend/app/ml/yolo_predictor.py))
  auto-detects the model task. If `quality_classifier.pt` is absent it falls
  back to the legacy YOLOv8 **detection** model (`best_damage_model.pt`), which
  also yields damage type / location / severity but was trained on only ~45
  images (low accuracy — see [`ml/EXECUTIVE_SUMMARY.txt`](ml/EXECUTIVE_SUMMARY.txt)).

## Configuration (backend `.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `ML_ENABLED` | `true` | Use the real YOLOv8 model (falls back to placeholder on error). |
| `ML_MODEL_PATH` | *(auto)* | Override path to the `.pt` weights. |
| `ML_CONF_THRESHOLD` | `0.25` | YOLO confidence threshold. |
| `ML_IOU_THRESHOLD` | `0.5` | YOLO IoU threshold. |
| `SECRET_KEY` | `change-me-in-production` | JWT signing secret. |
| `DATABASE_URL` | `sqlite:///./qualityvision.db` | Database URL. |
