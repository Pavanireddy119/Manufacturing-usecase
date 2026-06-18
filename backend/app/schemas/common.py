from pydantic import BaseModel, ConfigDict


class MessageResponse(BaseModel):
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"message": "User registered successfully"},
        }
    )


class ErrorResponse(BaseModel):
    success: bool = False
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"success": False, "message": "Error message"},
        }
    )


class HealthResponse(BaseModel):
    status: str

    model_config = ConfigDict(json_schema_extra={"example": {"status": "healthy"}})
