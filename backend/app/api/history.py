from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.prediction import PredictionHistoryResponse
from app.services.history_service import HistoryService

router = APIRouter(prefix="/api", tags=["Prediction History"])


@router.get(
    "/history",
    response_model=list[PredictionHistoryResponse],
    summary="List prediction history",
    description="Return all prediction records for the current user, newest first, including the stored image path for frontend display.",
    responses={
        200: {
            "description": "Prediction history for current user",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": 1,
                            "image_name": "mirror01.jpg",
                            "stored_image_path": "uploads/mirror01.jpg",
                            "prediction": "Defective",
                            "confidence_score": 96.2,
                            "created_at": "2026-06-17T12:00:00Z",
                        }
                    ]
                }
            },
        },
        401: {"description": "Invalid or missing token"},
    },
)
def get_history(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> list[PredictionHistoryResponse]:
    records = HistoryService(db).list_predictions(current_user)
    return [PredictionHistoryResponse.model_validate(record) for record in records]
