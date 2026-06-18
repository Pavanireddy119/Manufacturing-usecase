"""End-to-end smoke test for the integrated QualityVision AI backend.

Exercises signup -> login -> upload -> predict -> history against the in-process
ASGI app via FastAPI's TestClient. Uses a throwaway SQLite DB and a real sample
image from the ml/ dataset. Run from the backend/ directory:

    .venv\\Scripts\\python.exe test_integration_e2e.py
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

# Isolate test state from any real dev database / uploads.
_tmp = Path(tempfile.mkdtemp(prefix="qvai_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp / 'test.db').as_posix()}"
os.environ["UPLOAD_DIR"] = str(_tmp / "uploads")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = REPO_ROOT / "ml" / "dataset_final" / "images" / "test" / "0003.JPEG"


def main() -> int:
    assert SAMPLE_IMAGE.exists(), f"sample image missing: {SAMPLE_IMAGE}"
    # Context manager runs the lifespan startup (create_tables()).
    with TestClient(app) as client:
        return _run(client)


def _run(client) -> int:
    # Health
    r = client.get("/health")
    assert r.status_code == 200, r.text
    print("health:", r.json())

    # Signup
    username = f"tester_{uuid.uuid4().hex[:8]}"
    email = f"{username}@example.com"
    password = "Str0ngP@ss!"
    r = client.post(
        "/api/auth/signup",
        json={"username": username, "email": email, "password": password},
    )
    assert r.status_code in (200, 201), f"signup failed: {r.status_code} {r.text}"
    print("signup:", r.status_code)

    # Login
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("login: token acquired")

    # Upload
    with SAMPLE_IMAGE.open("rb") as fh:
        r = client.post(
            "/api/upload",
            headers=headers,
            files={"file": ("sample.jpg", fh, "image/jpeg")},
        )
    assert r.status_code in (200, 201), f"upload failed: {r.status_code} {r.text}"
    image_id = r.json()["image_id"]
    print("upload:", r.json())

    # Predict
    r = client.post(f"/api/predict/{image_id}", headers=headers)
    assert r.status_code == 200, f"predict failed: {r.status_code} {r.text}"
    pred = r.json()
    print("predict:", pred)
    assert pred["prediction"] in ("Defective", "Non-Defective")
    assert isinstance(pred["confidence_score"], (int, float))

    # History
    r = client.get("/api/history", headers=headers)
    assert r.status_code == 200, f"history failed: {r.status_code} {r.text}"
    hist = r.json()
    print("history count:", len(hist if isinstance(hist, list) else hist.get("items", [])))

    print("\nE2E OK — prediction served by:",
          "REAL YOLOv8 model" if pred.get("damage_type") is not None or pred["confidence_score"] == 95.0
          else "predictor (placeholder or no-damage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
