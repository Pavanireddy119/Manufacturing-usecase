from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.upload import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/api", tags=["Image Upload"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an inspection image",
    description="Upload one car light or mirror image, store it once, and return the generated image ID for prediction.",
    responses={
        201: {
            "description": "Image uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "image_id": 1,
                        "image_name": "uuid-file.jpg",
                        "image_path": "uploads/uuid-file.jpg",
                    }
                }
            },
        },
        400: {"description": "Unsupported or empty file"},
        401: {"description": "Invalid or missing token"},
    },
)
async def upload_image(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[
        UploadFile,
        File(description="JPG, JPEG, or PNG image to store for later prediction."),
    ],
) -> UploadResponse:
    uploaded_image = await UploadService(db).save_upload(file=file, user=current_user)
    return UploadResponse(
        success=True,
        image_id=uploaded_image.id,
        image_name=uploaded_image.image_name,
        image_path=uploaded_image.image_path,
    )
