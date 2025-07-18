import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras import layers

# Load the model
Model = tf.keras.models.load_model("C:/Users/Moonwalking/potato-disease-classification/saved_models/1/potatoes.h5", compile=False)


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Streamlit UI
st.title("Potato Disease Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # Ensure RGB
    image = image.resize((256, 256))  # Resize to match model input
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize (0 to 1)
    return image

if uploaded_file is not None:
    image = read_file_as_image(uploaded_file.getvalue())
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get raw model prediction
    prediction = Model(img_batch)  # No need for dictionary key
    prediction = tf.nn.softmax(prediction).numpy()  # Apply softmax for probabilities

    st.write("Raw Predictions:", prediction)  # Debugging

    predicted_class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    st.success(f"Prediction: {predicted_class} ({confidence:.2f}%)")
