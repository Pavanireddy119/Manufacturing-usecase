# QualityVision AI Backend

FastAPI backend for QualityVision AI, an AI-powered manufacturing quality inspection application for car lights and car mirrors.

## Features

- FastAPI with Swagger at `/docs` and ReDoc at `/redoc`
- SQLAlchemy 2.0 ORM with SQLite
- JWT authentication with bcrypt password hashing
- Protected image upload API
- Prediction by existing uploaded `image_id`
- Replaceable ML `Predictor` abstraction
- Prediction history per authenticated user
- CORS for React and Vite development origins
- Consistent JSON error responses

## Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

The API will be available at:

- Health: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Environment Variables

| Name | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | `change-me-in-production` | JWT signing secret. Replace in real deployments. |
| `ALGORITHM` | `HS256` | JWT signing algorithm. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | Access token lifetime. |
| `DATABASE_URL` | `sqlite:///./qualityvision.db` | SQLAlchemy database URL. |
| `BACKEND_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | Comma-separated allowed frontend origins. |

## API Flow

1. Register with `POST /api/auth/signup`.
2. Login with `POST /api/auth/login`.
3. Authorize in Swagger using `Bearer <access_token>`.
4. Upload an image with `POST /api/upload`.
5. Predict using the returned `image_id` with `POST /api/predict/{image_id}`.
6. View results with `GET /api/history`.

## ML Integration

The Sprint 1 ML implementation is intentionally isolated:

- `app/ml/model_loader.py` is the future TensorFlow/Keras loading point.
- `app/ml/predictor.py` exposes the `Predictor` class used by services.
- `app/ml/inference.py` currently returns randomized placeholder predictions.

ML engineers can replace the predictor implementation later without changing API routes, services, schemas, or database models.

## Project Structure

```text
backend/
  app/
    api/
    core/
    ml/
    models/
    schemas/
    services/
    main.py
  uploads/
  .env.example
  requirements.txt
  README.md
```
