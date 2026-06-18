import random

from app.schemas.prediction import PredictionResult


def run_placeholder_inference(image_path: str) -> PredictionResult:
    """Deterministic-shape fallback used when the real ML model is unavailable.

    Returns a randomized binary verdict so the API stays functional even
    without ``torch`` / ``ultralytics`` installed or the model weights present.
    """
    prediction = random.choice(["Defective", "Non-Defective"])
    confidence_score = round(random.uniform(85.0, 99.5), 2)
    return PredictionResult(
        prediction=prediction,
        confidence_score=confidence_score,
    )
