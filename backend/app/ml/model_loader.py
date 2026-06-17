import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelLoader:
    """TensorFlow/Keras model loading integration point for future sprints."""

    def __init__(self) -> None:
        self._model: Any | None = None

    def load_model(self) -> Any | None:
        logger.info("Using Sprint 1 placeholder ML model")
        return self._model
