"""Real ML inference adapter.

Bridges the FastAPI backend to the YOLOv8 model in the ``ml/`` package and maps
its output onto the backend's binary quality contract
(``Defective`` / ``Non-Defective``), with optional damage detail.

Two model types are supported transparently:

* **Classification** (default): a YOLOv8-cls model trained on damage-vs-whole
  vehicle images. Accurate binary quality verdict from the top-1 class.
* **Detection**: the original YOLOv8 damage-detection pipeline (kept for
  backwards compatibility); also yields damage type / location / severity.

The backend never imports heavy ML libraries at module load time; everything is
imported lazily inside :class:`YoloDamagePredictor` so the API still boots if
``torch`` / ``ultralytics`` are unavailable, raising a recoverable error the
caller falls back from.
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
# Prefer the accurate binary classifier; fall back to the detection model.
CLASSIFIER_PATH = ML_PACKAGE_DIR / "quality_classifier.pt"
DETECTION_PATH = ML_PACKAGE_DIR / "best_damage_model.pt"
DEFAULT_MODEL_PATH = CLASSIFIER_PATH if CLASSIFIER_PATH.exists() else DETECTION_PATH

# Class-name fragments that indicate a defective/damaged sample.
_DEFECT_KEYWORDS = ("damage", "defect", "broken", "crack", "dent", "scratch", "bad")


class MLUnavailableError(RuntimeError):
    """Raised when the real ML stack cannot be loaded (missing deps or weights)."""


class YoloDamagePredictor:
    """Lazy, cached wrapper around the YOLOv8 quality model."""

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
        self._task: str | None = None
        self._pipeline: Any | None = None

    def _load(self) -> None:
        """Load the YOLO weights once and detect the model task."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise MLUnavailableError(f"Model weights not found at {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # torch / ultralytics missing
            raise MLUnavailableError(f"ML dependencies unavailable: {exc}") from exc

        self._model = YOLO(str(self.model_path))
        self._task = getattr(self._model, "task", None)
        logger.info("Loaded YOLOv8 %s model from %s", self._task, self.model_path)

    def predict(self, image_path: str) -> dict[str, Any]:
        """Run inference on a single image and return a normalized result dict.

        Keys: ``prediction``, ``confidence_score``, ``damage_type``,
        ``damage_location``, ``severity``, ``recommendation``.

        Raises :class:`MLUnavailableError` if the ML stack cannot run; callers
        are expected to fall back to the placeholder predictor.
        """
        self._load()
        if self._task == "classify":
            return self._predict_classify(image_path)
        return self._predict_detect(image_path)

    # -- classification ----------------------------------------------------
    def _predict_classify(self, image_path: str) -> dict[str, Any]:
        results = self._model.predict(image_path, imgsz=160, verbose=False)
        result = results[0]
        probs = result.probs
        top1 = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = str(result.names.get(top1, top1)).lower()

        is_defective = any(key in class_name for key in _DEFECT_KEYWORDS)
        if is_defective:
            return {
                "prediction": "Defective",
                "confidence_score": round(confidence * 100, 2),
                "damage_type": "damage",
                "damage_location": None,
                "severity": None,
                "recommendation": "Manual quality review required",
            }
        return {
            "prediction": "Non-Defective",
            "confidence_score": round(confidence * 100, 2),
            "damage_type": None,
            "damage_location": None,
            "severity": None,
            "recommendation": "No repair required",
        }

    # -- detection (legacy pipeline) ---------------------------------------
    def _predict_detect(self, image_path: str) -> dict[str, Any]:
        pipeline = self._load_detection_pipeline()

        from PIL import Image

        image = Path(image_path)
        with Image.open(image) as img:
            image_size = img.size

        raw = pipeline.run_yolo_detection(self._model, image, conf=self.conf, iou=self.iou)
        detections = pipeline.enrich_detections(raw, image_size)
        response = pipeline.create_api_response(detections, image)

        if response.get("damage_detected"):
            return {
                "prediction": "Defective",
                "confidence_score": round(float(response.get("confidence") or 0.0) * 100, 2),
                "damage_type": response.get("damage_type"),
                "damage_location": response.get("damage_location"),
                "severity": response.get("severity"),
                "recommendation": response.get("recommendation"),
            }
        return {
            "prediction": "Non-Defective",
            "confidence_score": 95.0,
            "damage_type": None,
            "damage_location": None,
            "severity": None,
            "recommendation": response.get("recommendation") or "No repair required",
        }

    def _load_detection_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        pkg_dir = str(ML_PACKAGE_DIR)
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        try:
            import inspection_pipeline as pipeline  # type: ignore
        except ImportError as exc:
            raise MLUnavailableError(f"Detection pipeline unavailable: {exc}") from exc
        self._pipeline = pipeline
        return pipeline
