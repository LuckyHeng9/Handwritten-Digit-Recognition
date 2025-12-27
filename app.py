import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = "digit_cnn_model.h5"
IMG_SIZE = 28
CANVAS_SIZE = 280
STROKE_WIDTH = 10

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Please run train_model.py first.")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# UI
# =========================
st.title("✍️ Handwritten Digit Recognition")
st.write("Draw a digit **(0–9)** and click **Predict**")

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,1)",
    stroke_width=STROKE_WIDTH,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=CANVAS_SIZE,
    width=CANVAS_SIZE,
    drawing_mode="freedraw",
    key="canvas",
)

# =========================
# PREPROCESSING (MNIST STYLE)
# =========================
def preprocess_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Invert colors (white digit on black background)
    image = ImageOps.invert(image)
    # Resize smoothly
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    # Normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # Reshape for CNN
    img_array = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img_array

# =========================
# SHOW DRAWING
# =========================
# if canvas_result.image_data is not None:
#     st.subheader("📸 Your Drawing")
#     drawing = Image.fromarray(
#         canvas_result.image_data.astype("uint8"),
#         mode="RGBA"
#     )
#     st.image(drawing, caption="Canvas Drawing", width=280)

# =========================
# PREDICT
# =========================
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8"),
            mode="RGBA"
        ).convert("RGB")
        
        processed_img = preprocess_image(img)
        
        prediction = model.predict(processed_img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f"🧠 Prediction: **{digit}**")
        st.write(f"🎯 Confidence: **{confidence:.2f}%**")
    else:
        st.warning("⚠️ Please draw a digit first.")