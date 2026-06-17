import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import auth, history, predict, upload
from app.core.config import get_settings
from app.core.database import create_tables
from app.core.logging import configure_logging
from app.schemas.common import HealthResponse

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    create_tables()
    logger.info("Application startup complete")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered manufacturing quality inspection backend for car lights and mirrors.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Service health and readiness checks."},
        {"name": "Authentication", "description": "Signup, login, and current-user APIs."},
        {"name": "Image Upload", "description": "Upload and persist inspection images."},
        {"name": "Prediction", "description": "Run ML quality prediction against stored images."},
        {"name": "Prediction History", "description": "Retrieve past prediction results."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "message": str(exc.detail)},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message = "Validation error"
    if exc.errors():
        first_error = exc.errors()[0]
        location = ".".join(str(part) for part in first_error.get("loc", []))
        message = f"{location}: {first_error.get('msg', message)}"
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "message": message},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "message": "Internal server error"},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Public endpoint used by local development, load balancers, and deployment monitors to confirm the API is running.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {"application/json": {"example": {"status": "healthy"}}},
        }
    },
)
def health_check() -> HealthResponse:
    return HealthResponse(status="healthy")


app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(predict.router)
app.include_router(history.router)
