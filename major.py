import streamlit as st
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError as e:
    st.error(f"Error: {e}. Please ensure TensorFlow is installed.")
    st.stop()

import numpy as np
from PIL import Image
from io import BytesIO

# Load the pre-trained model (Make sure to update the path to your actual model)
MODEL_PATH = 'copy_move_forgery_model.h5'

# Ensure the model is loaded only once to avoid delays during every prediction
@st.cache_resource
def load_my_model():
    model = load_model(MODEL_PATH)
    return model

model = load_my_model()

# Function to process the uploaded image
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the model's input size (example: 224x224)
    
    # Convert the image to a numpy array and normalize it
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to detect copy-move forgery
def detect_forgery(model, img_array):
    prediction = model.predict(img_array)
    return prediction

# Streamlit App UI
st.title("Copy-Move Forgery Detection Using Deep Learning")
st.write("Upload an image to detect copy-move forgery.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = process_image(uploaded_file)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    st.write("Classifying... Please wait...")

    # Get prediction from the model
    prediction = detect_forgery(model, img_array)

    if prediction[0] > 0.5:  # Example threshold; adjust based on your model's output
        st.write("Forgery Detected!")
    else:
        st.write("No Forgery Detected!")
