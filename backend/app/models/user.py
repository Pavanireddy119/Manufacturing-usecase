from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.prediction import PredictionHistory
    from app.models.uploaded_image import UploadedImage


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    uploaded_images: Mapped[list["UploadedImage"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["PredictionHistory"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
