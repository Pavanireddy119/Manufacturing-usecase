import logging
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ml.predictor import Predictor
from app.models.prediction import PredictionHistory
from app.models.uploaded_image import UploadedImage
from app.models.user import User
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, db: Session, predictor: Predictor | None = None) -> None:
        self.db = db
        self.predictor = predictor or Predictor()

    def predict_uploaded_image(self, image_id: int, user: User) -> PredictionResponse:
        uploaded_image = self.db.scalar(
            select(UploadedImage).where(
                UploadedImage.id == image_id,
                UploadedImage.user_id == user.id,
            )
        )
        if uploaded_image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Uploaded image not found",
            )

        if not Path(uploaded_image.image_path).exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Stored image file not found",
            )

        result = self.predictor.predict(uploaded_image.image_path)
        history = PredictionHistory(
            user_id=user.id,
            image_name=uploaded_image.image_name,
            stored_image_path=uploaded_image.image_path,
            prediction=result.prediction,
            confidence_score=result.confidence_score,
        )
        self.db.add(history)
        self.db.commit()
        self.db.refresh(history)
        logger.info(
            "Prediction completed",
            extra={
                "user_id": user.id,
                "image_id": image_id,
                "prediction_id": history.id,
                "prediction": result.prediction,
            },
        )

        return PredictionResponse(
            success=True,
            image_id=image_id,
            image_name=uploaded_image.image_name,
            stored_image_path=uploaded_image.image_path,
            prediction=result.prediction,
            confidence_score=result.confidence_score,
            damage_type=result.damage_type,
            damage_location=result.damage_location,
            severity=result.severity,
            recommendation=result.recommendation,
        )
