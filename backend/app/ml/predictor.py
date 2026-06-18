import logging

from app.core.config import get_settings
from app.ml.inference import run_placeholder_inference
from app.ml.model_loader import ModelLoader
from app.ml.yolo_predictor import MLUnavailableError, YoloDamagePredictor
from app.schemas.prediction import PredictionResult

logger = logging.getLogger(__name__)


class Predictor:
    """Quality predictor.

    Prefers the real YOLOv8 vehicle-damage model and gracefully degrades to the
    placeholder predictor when the ML stack or weights are unavailable, so the
    API never fails to start or respond.
    """

    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self.model_loader = model_loader or ModelLoader()
        self.model = self.model_loader.load_model()
        self._settings = get_settings()
        self._yolo: YoloDamagePredictor | None = None
        self._ml_disabled = not self._settings.ml_enabled

        if self._settings.ml_enabled:
            self._yolo = YoloDamagePredictor(
                model_path=self._settings.ml_model_path or None,
                conf=self._settings.ml_conf_threshold,
                iou=self._settings.ml_iou_threshold,
            )

    def predict(self, image_path: str) -> PredictionResult:
        if self._yolo is not None and not self._ml_disabled:
            try:
                result = self._yolo.predict(image_path)
                return PredictionResult(**result)
            except MLUnavailableError as exc:
                # Disable further attempts; weights or deps are missing.
                logger.warning("Real ML model unavailable, using placeholder: %s", exc)
                self._ml_disabled = True
            except Exception:  # pragma: no cover - defensive runtime guard
                logger.exception("ML inference failed, using placeholder for this request")

        return run_placeholder_inference(image_path)
