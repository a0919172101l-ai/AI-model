from pathlib import Path
import joblib
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog

MODEL_PATH = Path("model.joblib")
VALID_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

st.set_page_config(page_title="Semiconductor AI Quality Check", page_icon="🔬")
st.title("Semiconductor AI Quality Inspection")
st.write(
    "Upload a product-surface image to test a traditional machine-learning prototype. "
    "The model predicts **Normal** or **Defective**."
)
st.caption("Educational prototype only — not certified for real semiconductor manufacturing.")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def image_to_features(image: Image.Image, image_size):
    image = image.convert("L").resize(tuple(image_size))
    pixels = np.asarray(image)
    features = hog(
        pixels,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )
    return features.reshape(1, -1).astype(np.float32)

if not MODEL_PATH.exists():
    st.warning("No trained model found yet.")
    st.markdown(
        "1. Put labeled images in `data/Normal/` and `data/Defective/`.  \n"
        "2. Run `python train_model.py` in the project folder.  \n"
        "3. Run `streamlit run app.py` again."
    )
    st.stop()

try:
    bundle = load_model()
    model = bundle["model"]
    classes = bundle["classes"]
    image_size = bundle["image_size"]
except Exception as exc:
    st.error(f"Could not load model.joblib: {exc}")
    st.stop()

uploaded = st.file_uploader("Choose an image", type=VALID_TYPES)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please try another file.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Inspect Image", type="primary"):
        features = image_to_features(image, image_size)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        score = float(probabilities[list(model.classes_).index(prediction)])

        if prediction == "Normal":
            st.success("Prediction: Normal")
        else:
            st.error("Prediction: Defective")

        st.metric("Model confidence estimate", f"{score * 100:.1f}%")
        st.caption(
            "This percentage is the model's estimated class probability, not a guarantee "
            "that the prediction is correct. Performance depends on training data."
        )
        st.write("Class probability estimates:")
        for label, probability in zip(model.classes_, probabilities):
            st.write(f"- {label}: {probability * 100:.1f}%")

st.divider()
st.subheader("How the prototype works")
st.write("Image upload → grayscale resize → HOG feature extraction → SVM classification → result")
st.write(
    "**Business value:** an early screening aid that may help organize quality checks. "
    "Real deployment would require representative factory data, validation, and expert review."
)

