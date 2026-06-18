"""Real ML inference adapter.

Bridges the FastAPI backend to the YOLOv8 vehicle-damage detection pipeline that
lives in the ``ml/`` package (produced by the data-preprocessing branch).

The backend never imports heavy ML libraries at module load time. Everything is
imported lazily inside :class:`YoloDamagePredictor` so that:

* the API server still boots if ``torch`` / ``ultralytics`` are not installed, and
* a clear, recoverable error is raised that the caller can fall back from.

The pipeline produces a rich damage report; this adapter maps it onto the
backend's binary quality contract (``Defective`` / ``Non-Defective``) while also
surfacing the richer damage details as optional fields.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# backend/app/ml/yolo_predictor.py -> repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
ML_PACKAGE_DIR = REPO_ROOT / "ml" / "integration_package"
DEFAULT_MODEL_PATH = ML_PACKAGE_DIR / "best_damage_model.pt"


class MLUnavailableError(RuntimeError):
    """Raised when the real ML stack cannot be loaded (missing deps or weights)."""


class YoloDamagePredictor:
    """Lazy, cached wrapper around the YOLOv8 inspection pipeline."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.conf = conf
        self.iou = iou
        self._model: Any | None = None
        self._pipeline: Any | None = None

    def _load(self) -> None:
        """Import the ml pipeline and load the YOLO weights once."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise MLUnavailableError(f"Model weights not found at {self.model_path}")

        # Make the self-contained ml package importable without installing it.
        pkg_dir = str(ML_PACKAGE_DIR)
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)

        try:
            import inspection_pipeline as pipeline  # type: ignore
        except ImportError as exc:  # ultralytics / torch / reportlab missing
            raise MLUnavailableError(f"ML dependencies unavailable: {exc}") from exc

        self._pipeline = pipeline
        self._model = pipeline.load_model(str(self.model_path))
        logger.info("Loaded YOLOv8 damage model from %s", self.model_path)

    def predict(self, image_path: str) -> dict[str, Any]:
        """Run detection on a single image and return a normalized result dict.

        Returns keys: ``prediction``, ``confidence_score``, ``damage_type``,
        ``damage_location``, ``severity``, ``recommendation``.

        Raises :class:`MLUnavailableError` if the ML stack cannot run; callers
        are expected to fall back to the placeholder predictor.
        """
        self._load()
        pipeline = self._pipeline
        assert pipeline is not None  # narrowed by _load()

        from PIL import Image

        image = Path(image_path)
        with Image.open(image) as img:
            image_size = img.size

        raw = pipeline.run_yolo_detection(self._model, image, conf=self.conf, iou=self.iou)
        detections = pipeline.enrich_detections(raw, image_size)
        response = pipeline.create_api_response(detections, image)

        damage_detected = bool(response.get("damage_detected"))
        confidence = float(response.get("confidence") or 0.0)

        if damage_detected:
            prediction = "Defective"
            confidence_score = round(confidence * 100, 2)
        else:
            prediction = "Non-Defective"
            # No detection => report inspection pass confidence.
            confidence_score = 95.0

        return {
            "prediction": prediction,
            "confidence_score": confidence_score,
            "damage_type": response.get("damage_type"),
            "damage_location": response.get("damage_location"),
            "severity": response.get("severity"),
            "recommendation": response.get("recommendation"),
        }
