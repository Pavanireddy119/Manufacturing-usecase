from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SignupRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50, examples=["admin"])
    email: EmailStr = Field(examples=["admin@test.com"])
    password: str = Field(min_length=8, examples=["Password123!"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "admin",
                "email": "admin@test.com",
                "password": "Password123!",
            }
        }
    )


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, examples=["admin"])
    password: str = Field(min_length=8, examples=["Password123!"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "username": "admin",
                "password": "Password123!",
            }
        }
    )


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "jwt-token",
                "token_type": "bearer",
            }
        }
    )


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "username": "admin",
                "email": "admin@test.com",
                "created_at": "2026-06-17T12:00:00Z",
            }
        },
    )
