"""
Streamlit inspection dashboard.

Run:
    streamlit run dashboard_app.py
"""

from pathlib import Path

import streamlit as st

try:
    from .inspection_pipeline import inspect_uploaded_bytes
except ImportError:
    from inspection_pipeline import inspect_uploaded_bytes


st.set_page_config(page_title="Vehicle Inspection Dashboard", layout="wide")

st.title("Vehicle Inspection Dashboard")

with st.sidebar:
    st.header("Inspection Controls")
    model_path = st.text_input("YOLO model path", value="")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.90, 0.50, 0.05)

uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded:
    image_bytes = uploaded.getvalue()
    with st.spinner("Running visual inspection..."):
        report = inspect_uploaded_bytes(
            image_bytes,
            uploaded.name,
            model_path=model_path or None,
            output_dir="../outputs/inspection_reports",
            conf=confidence,
            iou=iou,
        )

    left, right = st.columns([1.2, 1])

    with left:
        annotated_path = Path(report["artifacts"]["annotated_image"])
        st.image(str(annotated_path), caption="Annotated Image", use_container_width=True)

    with right:
        status = report["inspection_status"]
        st.subheader("Inspection Result")
        st.metric("Result", status)
        st.metric("Confidence", f"{report['confidence'] * 100:.1f}%")

        st.divider()
        st.write("**Damage Type**")
        st.write(report["damage_type"] or "None")
        st.write("**Location**")
        st.write(report["damage_location"] or "None")
        st.write("**Severity**")
        st.write(report["severity"] or "None")
        st.write("**Repair Recommendation**")
        st.write(report["recommendation"])

        st.divider()
        st.download_button(
            "Download JSON Report",
            data=Path(report["artifacts"]["inspection_report_json"]).read_bytes(),
            file_name="inspection_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download PDF Report",
            data=Path(report["artifacts"]["inspection_report_pdf"]).read_bytes(),
            file_name="inspection_report.pdf",
            mime="application/pdf",
        )

    st.subheader("Detections")
    if report["detections"]:
        st.dataframe(report["detections"], use_container_width=True)
    else:
        st.info("No damage detected.")
else:
    st.info("Upload a vehicle image to start inspection.")
