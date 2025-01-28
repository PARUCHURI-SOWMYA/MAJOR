import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import match_template

# Function to process the uploaded image (convert to grayscale for simpler processing)
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the processing size (224x224)
    
    # Convert to numpy array
    img_array = np.array(img)
    return img_array

# Function to perform simple copy-move forgery detection using template matching
def detect_copy_move_forgery(img_array):
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    # For simplicity, let's use template matching (this can be more complex for actual copy-move forgery)
    # We split the image into patches and check for duplicated patches
    patch_size = 32  # Patch size for detection
    height, width = gray_img.shape
    
    for i in range(0, height - patch_size, patch_size):
        for j in range(0, width - patch_size, patch_size):
            # Extract a patch
            patch = gray_img[i:i + patch_size, j:j + patch_size]
            
            # Try matching this patch within the rest of the image (template matching)
            result = match_template(gray_img, patch)
            # Check if a match is found
            threshold = 0.9  # Adjust this threshold based on performance needs
            if np.any(result >= threshold):
                return True  # Match found, indicating possible forgery
    return False  # No matches found, no forgery detected

# Streamlit App UI
st.title("Copy-Move Forgery Detection Using OpenCV")
st.write("Upload an image to detect copy-move forgery.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = process_image(uploaded_file)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    st.write("Classifying... Please wait...")

    # Get prediction from the forgery detection function
    is_forged = detect_copy_move_forgery(img_array)

    if is_forged:
        st.write("Forgery Detected!")
    else:
        st.write("No Forgery Detected!")
