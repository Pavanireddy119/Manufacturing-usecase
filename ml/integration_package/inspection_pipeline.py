"""
Manufacturing-grade vehicle visual inspection pipeline.

Stages:
Image Upload/Input -> YOLO Detection -> Location Classification ->
Severity Classification -> Business Rules -> Report/Artifact Generation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("output/ultralytics_config").resolve()))

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "ultralytics is required to run inspection. Install it with: pip install ultralytics"
    ) from exc


DAMAGE_TYPES = {"dent", "scratch", "crack", "broken_part", "paint_damage"}
DAMAGE_LOCATIONS = {
    "front_bumper",
    "rear_bumper",
    "hood",
    "windshield",
    "left_door",
    "right_door",
    "roof",
    "side_panel",
    "side_mirror",
}
SEVERITY_LEVELS = {"low", "medium", "high"}

MODEL_SEARCH_PATHS = [
    "output/yolo_fixed/weights/best.pt",
    "best_damage_model.pt",
    "output/models_fixed/best.pt",
    "output/models/weights/best.pt",
    "output/models/best.pt",
    "runs/detect/train/weights/best.pt",
    "runs/detect/train2/weights/best.pt",
]


@dataclass
class DamageDetection:
    damage_type: str
    damage_location: str
    severity: str
    confidence: float
    bounding_box: List[int]
    recommendation: str


def find_model_path(model_path: Optional[str] = None) -> Path:
    """Find the best available YOLO damage model."""
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model not found: {candidate}")

    for path in MODEL_SEARCH_PATHS:
        candidate = Path(path)
        if candidate.exists():
            return candidate

    best_matches = sorted(Path(".").glob("**/best.pt"))
    if best_matches:
        return best_matches[0]

    raise FileNotFoundError("No trained YOLO model found. Expected output/yolo_fixed/weights/best.pt.")


def load_model(model_path: Optional[str] = None) -> YOLO:
    return YOLO(str(find_model_path(model_path)))


def normalize_damage_type(class_name: str) -> Optional[str]:
    """Map model class labels into the approved business taxonomy."""
    normalized = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in DAMAGE_TYPES:
        return normalized
    if normalized == "other_damage":
        return None
    return normalized if normalized in DAMAGE_TYPES else None


def classify_location(bbox: Sequence[float], image_size: Sequence[int]) -> str:
    """
    Deterministic first-pass location classifier.

    This is intentionally isolated so a trained vehicle-part classifier or
    segmentation model can replace it without changing the API/report layer.
    """
    x1, y1, x2, y2 = bbox
    width, height = image_size
    cx = ((x1 + x2) / 2) / max(width, 1)
    cy = ((y1 + y2) / 2) / max(height, 1)
    box_w = (x2 - x1) / max(width, 1)
    box_h = (y2 - y1) / max(height, 1)

    if cy < 0.22:
        return "roof"
    if cy < 0.40:
        if 0.35 <= cx <= 0.65 and box_h > 0.10:
            return "windshield"
        return "hood"
    if cy > 0.76:
        return "front_bumper" if cx <= 0.55 else "rear_bumper"
    if box_w < 0.12 and 0.28 <= cy <= 0.62 and (cx < 0.25 or cx > 0.75):
        return "side_mirror"
    if cx < 0.35:
        return "left_door"
    if cx > 0.65:
        return "right_door"
    return "side_panel"


def estimate_severity(damage_type: str, bbox: Sequence[float], image_size: Sequence[int]) -> str:
    """Estimate severity from relative defect area and damage criticality."""
    x1, y1, x2, y2 = bbox
    width, height = image_size
    image_area = max(width * height, 1)
    area_ratio = max((x2 - x1) * (y2 - y1), 0) / image_area

    multiplier = {
        "scratch": 0.80,
        "paint_damage": 0.90,
        "dent": 1.10,
        "crack": 1.30,
        "broken_part": 1.60,
    }.get(damage_type, 1.0)
    score = area_ratio * multiplier

    if damage_type in {"broken_part"} and area_ratio > 0.015:
        return "high"
    if damage_type == "crack" and area_ratio > 0.010:
        return "high"
    if score >= 0.045:
        return "high"
    if score >= 0.012:
        return "medium"
    return "low"


def generate_recommendation(damage_type: str, location: str, severity: str) -> str:
    """Business rules engine for repair disposition."""
    if damage_type == "crack" and location == "windshield":
        return "Replace windshield"
    if damage_type == "broken_part":
        return "Component replacement required"
    if damage_type == "paint_damage":
        return "Repaint affected area"
    if damage_type == "dent" and severity == "high":
        return "Replace damaged panel"
    if damage_type == "dent":
        return "Minor repair required" if severity == "low" else "Panel repair required"
    if damage_type == "scratch":
        return "Polish and touch-up required" if severity == "low" else "Refinish affected panel"
    if damage_type == "crack":
        return "Repair or replace cracked component"
    return "Manual quality review required"


def run_yolo_detection(model: YOLO, image_path: Path, conf: float, iou: float) -> List[Dict[str, Any]]:
    results = model.predict(str(image_path), conf=conf, iou=iou, imgsz=640, verbose=False)
    if not results or results[0].boxes is None:
        return []

    detections: List[Dict[str, Any]] = []
    result = results[0]
    for idx in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[idx].item())
        class_name = result.names.get(cls_id, f"class_{cls_id}")
        damage_type = normalize_damage_type(class_name)
        if damage_type is None:
            continue
        detections.append(
            {
                "damage_type": damage_type,
                "confidence": float(result.boxes.conf[idx].item()),
                "bounding_box": result.boxes.xyxy[idx].tolist(),
            }
        )
    return detections


def enrich_detections(raw_detections: List[Dict[str, Any]], image_size: Sequence[int]) -> List[DamageDetection]:
    enriched = []
    for det in raw_detections:
        bbox = [int(round(value)) for value in det["bounding_box"]]
        damage_type = det["damage_type"]
        location = classify_location(bbox, image_size)
        severity = estimate_severity(damage_type, bbox, image_size)
        recommendation = generate_recommendation(damage_type, location, severity)
        enriched.append(
            DamageDetection(
                damage_type=damage_type,
                damage_location=location,
                severity=severity,
                confidence=round(det["confidence"], 4),
                bounding_box=bbox,
                recommendation=recommendation,
            )
        )
    return sorted(enriched, key=lambda item: item.confidence, reverse=True)


def create_api_response(
    detections: List[DamageDetection],
    image_path: Path,
    artifacts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).isoformat()
    primary = detections[0] if detections else None

    response: Dict[str, Any] = {
        "inspection_status": "FAILED" if primary else "PASSED",
        "damage_detected": bool(primary),
        "damage_type": primary.damage_type if primary else None,
        "damage_location": primary.damage_location if primary else None,
        "severity": primary.severity if primary else None,
        "confidence": primary.confidence if primary else 0.0,
        "bounding_box": primary.bounding_box if primary else None,
        "recommendation": primary.recommendation if primary else "No repair required",
        "timestamp": timestamp,
        "image": str(image_path),
        "detections": [asdict(det) for det in detections],
    }
    if artifacts:
        response["artifacts"] = artifacts
    return response


def draw_annotated_image(image_path: Path, detections: List[DamageDetection], output_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small_font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    severity_color = {
        "low": (30, 160, 70),
        "medium": (230, 150, 20),
        "high": (210, 45, 45),
    }

    for det in detections:
        x1, y1, x2, y2 = det.bounding_box
        color = severity_color.get(det.severity, (255, 110, 40))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        label_lines = [
            det.damage_type.replace("_", " ").title(),
            det.damage_location.replace("_", " ").title(),
            det.severity.title(),
            f"{det.confidence * 100:.0f}%",
        ]
        line_heights = []
        max_width = 0
        for line in label_lines:
            left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
            max_width = max(max_width, right - left)
            line_heights.append(bottom - top)
        label_h = sum(line_heights) + 10 + (len(label_lines) - 1) * 3
        label_w = max_width + 14
        label_y = max(0, y1 - label_h - 4)

        draw.rectangle([x1, label_y, x1 + label_w, label_y + label_h], fill=(20, 24, 28))
        cursor_y = label_y + 5
        for line in label_lines:
            draw.text((x1 + 7, cursor_y), line, fill=(255, 255, 255), font=font)
            cursor_y += line_heights.pop(0) + 3

    if not detections:
        draw.text((16, 16), "Inspection Passed - No Damage Detected", fill=(30, 160, 70), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)
    return output_path


def write_json_report(response: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    return output_path


def write_pdf_report(response: Dict[str, Any], annotated_image_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36)
    styles = getSampleStyleSheet()
    elements: List[Any] = [
        Paragraph("Vehicle Visual Inspection Report", styles["Title"]),
        Spacer(1, 12),
    ]

    summary = [
        ["Inspection Result", response["inspection_status"]],
        ["Damage Detected", str(response["damage_detected"])],
        ["Damage Type", str(response["damage_type"])],
        ["Location", str(response["damage_location"])],
        ["Severity", str(response["severity"])],
        ["Confidence", f"{response['confidence'] * 100:.1f}%"],
        ["Repair Recommendation", response["recommendation"]],
        ["Timestamp", response["timestamp"]],
    ]
    table = Table(summary, colWidths=[150, 330])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1f2937")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    elements.extend([table, Spacer(1, 18)])

    if Path(annotated_image_path).exists():
        elements.append(Paragraph("Annotated Image", styles["Heading2"]))
        elements.append(Spacer(1, 8))
        elements.append(PdfImage(str(annotated_image_path), width=480, height=320, kind="proportional"))

    doc.build(elements)
    return output_path


def inspect_image(
    image_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "output/inspection",
    conf: float = 0.25,
    iou: float = 0.5,
    model: Optional[YOLO] = None,
) -> Dict[str, Any]:
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    output = Path(output_dir)
    model = model or load_model(model_path)
    with Image.open(image) as img:
        image_size = img.size

    raw = run_yolo_detection(model, image, conf=conf, iou=iou)
    detections = enrich_detections(raw, image_size)

    stem = image.stem
    annotated_path = output / "annotated_image.jpg"
    json_path = output / "inspection_report.json"
    pdf_path = output / "inspection_report.pdf"
    per_image_annotated = output / f"{stem}_annotated.jpg"

    draw_annotated_image(image, detections, annotated_path)
    if per_image_annotated != annotated_path:
        draw_annotated_image(image, detections, per_image_annotated)

    artifacts = {
        "inspection_report_json": str(json_path),
        "inspection_report_pdf": str(pdf_path),
        "annotated_image": str(annotated_path),
        "annotated_image_named": str(per_image_annotated),
    }
    response = create_api_response(detections, image, artifacts)
    write_json_report(response, json_path)
    write_pdf_report(response, annotated_path, pdf_path)
    return response


def inspect_uploaded_bytes(
    image_bytes: bytes,
    filename: str,
    model_path: Optional[str] = None,
    output_dir: str = "output/inspection",
    conf: float = 0.25,
    iou: float = 0.5,
    model: Optional[YOLO] = None,
) -> Dict[str, Any]:
    suffix = Path(filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name
    return inspect_image(temp_path, model_path=model_path, output_dir=output_dir, conf=conf, iou=iou, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vehicle visual inspection on one image.")
    parser.add_argument("image", help="Path to vehicle image")
    parser.add_argument("--model", default=None, help="YOLO model path")
    parser.add_argument("--output-dir", default="output/inspection", help="Artifact output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    args = parser.parse_args()

    response = inspect_image(args.image, model_path=args.model, output_dir=args.output_dir, conf=args.conf, iou=args.iou)
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
