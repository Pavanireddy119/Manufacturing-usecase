from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PredictionResult(BaseModel):
    prediction: str
    confidence_score: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "Defective",
                "confidence_score": 96.25,
            }
        }
    )


class PredictionResponse(BaseModel):
    success: bool
    image_id: int
    image_name: str
    stored_image_path: str
    prediction: str
    confidence_score: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "image_id": 1,
                "image_name": "mirror01.jpg",
                "stored_image_path": "uploads/mirror01.jpg",
                "prediction": "Defective",
                "confidence_score": 96.25,
            }
        }
    )


class PredictionHistoryResponse(BaseModel):
    id: int
    image_name: str
    stored_image_path: str
    prediction: str
    confidence_score: float
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "image_name": "mirror01.jpg",
                "stored_image_path": "uploads/mirror01.jpg",
                "prediction": "Defective",
                "confidence_score": 96.2,
                "created_at": "2026-06-17T12:00:00Z",
            }
        },
    )
