from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.prediction import PredictionHistory
from app.models.user import User


class HistoryService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def list_predictions(self, user: User) -> list[PredictionHistory]:
        return list(
            self.db.scalars(
                select(PredictionHistory)
                .where(PredictionHistory.user_id == user.id)
                .order_by(PredictionHistory.created_at.desc())
            )
        )
