import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.uploaded_image import UploadedImage
from app.models.user import User

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


class UploadService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()

    async def save_upload(self, file: UploadFile, user: User) -> UploadedImage:
        original_name = file.filename or ""
        extension = self._validate_extension(original_name)
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        upload_dir = Path(self.settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_original = Path(original_name).name.replace(" ", "-")
        image_name = f"{uuid4()}-{safe_original}" if safe_original else f"{uuid4()}.{extension}"
        image_path = upload_dir / image_name
        image_path.write_bytes(contents)

        stored_path = image_path.as_posix()
        uploaded_image = UploadedImage(
            user_id=user.id,
            image_name=image_name,
            image_path=stored_path,
        )
        self.db.add(uploaded_image)
        self.db.commit()
        self.db.refresh(uploaded_image)
        logger.info(
            "Image uploaded",
            extra={"user_id": user.id, "image_id": uploaded_image.id, "image_path": stored_path},
        )
        return uploaded_image

    @staticmethod
    def _validate_extension(filename: str) -> str:
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Allowed formats: jpg, jpeg, png",
            )
        return extension
