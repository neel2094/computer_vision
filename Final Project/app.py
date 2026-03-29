import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2

from utils.detection import detect_pest, detect_disease, draw_boxes, crop_detections
from utils.classification import (
    classify_image_basic,
    classify_image_advanced
)

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Crop AI", layout="wide")

st.title("🌿 Smart Crop Monitoring System")
st.markdown("### Pest Detection • Disease Detection • AI Classification")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- MODE SELECT ----------------
mode = st.selectbox(
    "⚙️ Select Mode",
    [
        "Pest Detection",
        "Disease Detection",
        "Classification (Normal)",
        "Classification (Drone - Advanced)",
        "Full Pipeline (Detection + Classification)"
    ]
)

# ---------------- MAIN ----------------
if uploaded_file is not None:

    os.makedirs("uploads", exist_ok=True)
    image_path = os.path.join("uploads", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)
    image_cv = cv2.imread(image_path)

    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # ============================================================
    # 🐛 PEST DETECTION
    # ============================================================
    if mode == "Pest Detection":
        st.subheader("🐛 Pest Detection")

        with st.spinner("Detecting pests..."):
            results = detect_pest(image_path)
            output = draw_boxes(results)

        st.image(output, caption="Detection Result")

    # ============================================================
    # 🌿 DISEASE DETECTION
    # ============================================================
    elif mode == "Disease Detection":
        st.subheader("🌿 Disease Detection")

        with st.spinner("Detecting diseases..."):
            results = detect_disease(image_path)
            output = draw_boxes(results)

        st.image(output, caption="Detection Result")

    # ============================================================
    # 🧠 BASIC CLASSIFICATION
    # ============================================================
    elif mode == "Classification (Normal)":
        st.subheader("🧠 Basic Classification")

        with st.spinner("Classifying..."):
            label, conf = classify_image_basic(image)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.2f}")

    # ============================================================
    # 🚀 ADVANCED CLASSIFICATION (DRONE)
    # ============================================================
    elif mode == "Classification (Drone - Advanced)":
        st.subheader("🚀 Advanced Drone Classification")

        with st.spinner("Processing drone image..."):
            label, conf = classify_image_advanced(image_path)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {conf:.2f}")

    # ============================================================
    # 🔥 FULL PIPELINE
    # ============================================================
    elif mode == "Full Pipeline (Detection + Classification)":
        st.subheader("🔥 Full Pipeline: Detection → Classification")

        with st.spinner("Running detection..."):
            results = detect_disease(image_path)
            output = draw_boxes(results)

        st.image(output, caption="🔍 Detected Regions")

        crops = crop_detections(results, image_cv)

        if len(crops) == 0:
            st.warning("⚠️ No regions detected")
        else:
            st.success(f"Detected {len(crops)} regions")

            for i, crop in enumerate(crops):
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                col1, col2 = st.columns(2)

                with col1:
                    st.image(crop_pil, caption=f"Region {i+1}")

                with col2:
                    # 🔥 Use advanced classification here
                    label, conf = classify_image_advanced(image_path)

                    st.write(f"🧠 Prediction: {label}")
                    st.write(f"📊 Confidence: {conf:.2f}")

                st.markdown("---")