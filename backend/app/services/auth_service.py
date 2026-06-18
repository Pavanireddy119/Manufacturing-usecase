import logging

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, SignupRequest, TokenResponse

logger = logging.getLogger(__name__)


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def register_user(self, payload: SignupRequest) -> None:
        existing_username = self.db.scalar(
            select(User).where(User.username == payload.username)
        )
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        existing_email = self.db.scalar(select(User).where(User.email == payload.email))
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists",
            )

        user = User(
            username=payload.username,
            email=str(payload.email),
            password_hash=hash_password(payload.password),
        )
        self.db.add(user)
        self.db.commit()
        logger.info("User registered", extra={"username": payload.username})

    def authenticate_user(self, payload: LoginRequest) -> TokenResponse:
        user = self.db.scalar(select(User).where(User.username == payload.username))
        if user is None or not verify_password(payload.password, user.password_hash):
            logger.warning("Invalid login attempt", extra={"username": payload.username})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(subject=str(user.id))
        logger.info("User logged in", extra={"user_id": user.id})
        return TokenResponse(access_token=token, token_type="bearer")
