import random

from app.schemas.prediction import PredictionResult


def run_placeholder_inference(image_path: str) -> PredictionResult:
    prediction = random.choice(["Defective", "Non-Defective"])
    confidence_score = round(random.uniform(85.0, 99.5), 2)
    return PredictionResult(
        prediction=prediction,
        confidence_score=confidence_score,
    )
