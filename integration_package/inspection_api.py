"""
FastAPI service for vehicle visual inspection.

Run:
    uvicorn inspection_api:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

try:
    from .inspection_pipeline import inspect_uploaded_bytes
except ImportError:
    from inspection_pipeline import inspect_uploaded_bytes


app = FastAPI(title="Vehicle Visual Inspection API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    model_path: Optional[str] = Form(default=None),
    conf: float = Form(default=0.25),
    iou: float = Form(default=0.5),
) -> dict:
    image_bytes = await file.read()
    return inspect_uploaded_bytes(
        image_bytes,
        file.filename or "uploaded.jpg",
        model_path=model_path,
        output_dir="../outputs/inspection_reports",
        conf=conf,
        iou=iou,
    )


@app.get("/artifacts/{artifact_name}")
def get_artifact(artifact_name: str) -> FileResponse:
    allowed = {
        "inspection_report.json",
        "inspection_report.pdf",
        "annotated_image.jpg",
    }
    if artifact_name not in allowed:
        raise FileNotFoundError("Unsupported artifact")
    path = Path("../outputs/inspection_reports") / artifact_name
    return FileResponse(path)
