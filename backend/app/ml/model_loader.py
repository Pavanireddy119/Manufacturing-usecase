import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """Legacy placeholder-model hook.

    The real damage model is now loaded lazily by
    :class:`app.ml.yolo_predictor.YoloDamagePredictor`. This loader is retained
    only to back the placeholder fallback path.
    """

    def __init__(self) -> None:
        self._model: Any | None = None

    def load_model(self) -> Any | None:
        logger.info("Placeholder model hook initialized (real model handled by YoloDamagePredictor)")
        return self._model
