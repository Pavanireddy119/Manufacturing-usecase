"""Standalone check that the real YOLOv8 model loads and runs inference.

Exercises app.ml.yolo_predictor.YoloDamagePredictor (the same path the API
uses) on real sample images. Run from backend/:

    .venv\\Scripts\\python.exe test_model_inference.py
"""

import sys
from pathlib import Path

from app.ml.yolo_predictor import YoloDamagePredictor, DEFAULT_MODEL_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES = [
    REPO_ROOT / "ml" / "dataset_final" / "images" / "test" / "0003.JPEG",
    REPO_ROOT / "ml" / "dataset_final" / "images" / "test" / "0004.JPEG",
    REPO_ROOT / "ml" / "dataset_final" / "images" / "test" / "0005.JPEG",
]


def main() -> int:
    print("model path:", DEFAULT_MODEL_PATH, "exists:", DEFAULT_MODEL_PATH.exists())
    predictor = YoloDamagePredictor()
    ran = 0
    for img in SAMPLES:
        if not img.exists():
            print("skip (missing):", img)
            continue
        result = predictor.predict(str(img))
        ran += 1
        print(f"\n{img.name} ->")
        for k, v in result.items():
            print(f"    {k}: {v}")
        assert result["prediction"] in ("Defective", "Non-Defective")
        assert isinstance(result["confidence_score"], (int, float))
    assert ran > 0, "no sample images were available to test"
    print(f"\nMODEL OK — ran inference on {ran} image(s) without error.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
