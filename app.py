from pathlib import Path
import json
import numpy as np
import streamlit as st
from PIL import Image
from deepvision.classifier import load_classifier, predict_classifier
from deepvision.anomaly import AnomalyDetector, create_heatmap_overlay, list_images

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
CLASSIFIER_PATH = MODEL_DIR / "classifier.pt"
ANOMALY_CACHE = {}

st.set_page_config(page_title="DeepVision AI", page_icon="🔎", layout="wide")
st.title("🔎 DeepVision AI")
st.caption("Industrial visual inspection • Deep Learning classification + anomaly localization")
st.info(
    "This is an educational inspection prototype. Results depend on the training/reference images "
    "and are not a certified quality-control system."
)

with st.sidebar:
    st.header("Inspection settings")
    mode = st.radio(
        "Choose analysis mode",
        ["Combined inspection", "A — Defect classification", "B — Anomaly detection + heatmap"]
    )
    st.markdown("---")
    st.subheader("Quick setup")
    st.markdown(
        "1. Add labeled images to `data/classifier/` for classifier training.\n"
        "2. Add only good/normal reference images to `data/anomaly/reference_normal/`.\n"
        "3. Run `python scripts/train_classifier.py` for Mode A."
    )

uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read this image: {exc}")
        st.stop()

    col_img, col_result = st.columns([1, 1.2])
    with col_img:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    with col_result:
        st.subheader("Inspection results")
        if mode in ["Combined inspection", "A — Defect classification"]:
            if CLASSIFIER_PATH.exists():
                try:
                    model, classes, device = load_classifier(CLASSIFIER_PATH)
                    result = predict_classifier(model, classes, image, device)
                    st.markdown("#### A — Defect classification")
                    st.metric("Predicted class", result["label"])
                    st.metric("Model confidence", f'{result["confidence"]:.1%}')
                    st.caption("Class probabilities")
                    st.json({k: round(v, 4) for k, v in result["probabilities"].items()})
                except Exception as exc:
                    st.error(f"Classifier could not be loaded: {exc}")
            else:
                st.warning(
                    "Classifier model not found. Train it first with `python scripts/train_classifier.py`. "
                    "The anomaly-detection mode can still run independently."
                )

        if mode in ["Combined inspection", "B — Anomaly detection + heatmap"]:
            st.markdown("#### B — Anomaly detection")
            ref_dir = DATA_DIR / "anomaly" / "reference_normal"
            reference_paths = list_images(ref_dir)
            if len(reference_paths) < 2:
                st.warning(
                    f"Add at least 2 normal reference images to `{ref_dir.relative_to(ROOT)}`. "
                    "This mode compares the uploaded image with your own normal-product examples."
                )
            else:
                try:
                    with st.spinner("Building anomaly reference features (first run may take a while)..."):
                        detector = AnomalyDetector.from_reference_images(reference_paths)
                    anomaly = detector.score_image(image)
                    st.metric("Anomaly score (relative)", f'{anomaly["score"]:.3f}')
                    st.caption(
                        "Higher scores mean the image differs more from the reference set. "
                        "This is a relative score, not a calibrated probability."
                    )
                    st.image(
                        create_heatmap_overlay(image, anomaly["heatmap"]),
                        caption="Anomaly heatmap overlay (warmer regions indicate greater feature difference)",
                        use_container_width=True
                    )
                except Exception as exc:
                    st.error(
                        "Anomaly detection failed. Check that PyTorch/torchvision are installed and "
                        f"reference images are valid. Details: {exc}"
                    )
else:
    st.markdown("### What this project does")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A — Defect classification")
        st.write("A transfer-learning CNN predicts the trained label, for example `normal` or `defective`.")
        st.write("Requires labeled normal and defective training images.")
    with c2:
        st.subheader("B — Anomaly localization")
        st.write("A pretrained CNN compares local image features with normal reference examples and creates a heatmap.")
        st.write("Does not require defect labels, but does require normal reference images.")
    st.markdown("---")
    st.subheader("Project workflow")
    st.code("Image → preprocessing → CNN features → classification / anomaly scoring → result dashboard", language="text")
    st.markdown(
        "**Important:** do not interpret a confidence score or anomaly score as a guarantee. "
        "Use varied test images and report limitations during your presentation."
    )
