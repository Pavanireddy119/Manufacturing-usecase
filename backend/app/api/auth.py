from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, SignupRequest, TokenResponse, UserResponse
from app.schemas.common import MessageResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post(
    "/signup",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a QualityVision AI account with a unique username and email address.",
    responses={
        201: {"description": "User registered successfully"},
        400: {"description": "Username or email already exists"},
    },
)
def signup(payload: SignupRequest, db: Annotated[Session, Depends(get_db)]) -> MessageResponse:
    AuthService(db).register_user(payload)
    return MessageResponse(message="User registered successfully")


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login and receive a JWT",
    description="Authenticate with username and password. Returns a bearer token valid for 24 hours by default.",
    responses={
        200: {"description": "JWT token issued"},
        401: {"description": "Invalid credentials"},
    },
)
def login(payload: LoginRequest, db: Annotated[Session, Depends(get_db)]) -> TokenResponse:
    return AuthService(db).authenticate_user(payload)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Return profile details for the user represented by the current bearer token.",
    responses={
        200: {"description": "Current user details"},
        401: {"description": "Invalid or missing token"},
    },
)
def me(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    return current_user
