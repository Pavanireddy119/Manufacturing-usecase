from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post(
    "/predict/{image_id}",
    response_model=PredictionResponse,
    summary="Predict quality status for an uploaded image",
    description="Run the ML predictor against an image that was already uploaded with /api/upload. This endpoint does not upload or duplicate files.",
    responses={
        200: {
            "description": "Prediction completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "image_id": 1,
                        "image_name": "mirror01.jpg",
                        "stored_image_path": "uploads/mirror01.jpg",
                        "prediction": "Defective",
                        "confidence_score": 96.25,
                    }
                }
            },
        },
        401: {"description": "Invalid or missing token"},
        404: {"description": "Uploaded image not found"},
    },
)
def predict_image(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> PredictionResponse:
    return PredictionService(db).predict_uploaded_image(
        image_id=image_id,
        user=current_user,
    )
