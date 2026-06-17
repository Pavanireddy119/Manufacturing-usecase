from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UploadResponse(BaseModel):
    success: bool
    image_id: int
    image_name: str
    image_path: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "image_id": 1,
                "image_name": "8bf8fdbc-1ec9-4d1a-aed9-c8b906f9551f-car-light.jpg",
                "image_path": "uploads/8bf8fdbc-1ec9-4d1a-aed9-c8b906f9551f-car-light.jpg",
            }
        }
    )


class UploadedImageResponse(BaseModel):
    id: int
    image_name: str
    image_path: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
