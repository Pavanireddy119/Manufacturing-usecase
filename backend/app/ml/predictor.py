from app.ml.inference import run_placeholder_inference
from app.ml.model_loader import ModelLoader
from app.schemas.prediction import PredictionResult


class Predictor:
    def __init__(self, model_loader: ModelLoader | None = None) -> None:
        self.model_loader = model_loader or ModelLoader()
        self.model = self.model_loader.load_model()

    def predict(self, image_path: str) -> PredictionResult:
        return run_placeholder_inference(image_path)
