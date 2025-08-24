# ---------------- Ignore Warnings ----------------
import os, warnings, logging
import sys

# Silence TF backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hide INFO/WARN/ERROR, keep only FATAL
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # disable oneDNN logs
os.environ["KMP_WARNINGS"] = "0"             # silence OpenMP warnings

# Optionally suppress *all* stderr (kills SSE/AVX messages too)
class DevNull:
    def write(self, msg): pass
    def flush(self): pass
sys.stderr = DevNull()

# Suppress protobuf + python-side logs
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



# ---------------- Start ----------------
import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# ---------------- Load Model ----------------
model = keras.models.load_model("breed_classifier5.keras")

# Class names (must match training order!)
class_names = ['Beagle', 'Bengal', 'Bombay', 'British_Shorthair', 'German_Shepherd', 'Golden_Retriever', 'Persian', 'Pug', 'Siamese', 'Siberian_Husky']

# ---------------- Prediction Function ----------------
def predict_breed(img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    return predicted_class, confidence


# ---------------- Streamlit App ----------------
st.set_page_config(page_title="Animal Breed Classifier", layout="wide")

# Initialize mode
if "mode" not in st.session_state:
    st.session_state.mode = "upload"
    st.session_state.file = None

# ------------- Upload View -------------
if st.session_state.mode == "upload":
    st.title("🐾 Animal Breed Classifier")
    st.write("Upload an image of a dog or cat and let the AI predict its breed!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.file = uploaded_file
        st.session_state.mode = "result"
        st.rerun()

# ------------- Result View -------------
elif st.session_state.mode == "result":
    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state.file, caption="Uploaded Image", width=300)

    with col2:
        predicted_class, confidence = predict_breed(st.session_state.file)
        st.success(f"**Breed:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        if st.button("⬅ Upload another image"):
            st.session_state.mode = "upload"
            st.session_state.file = None
            st.rerun()

    
