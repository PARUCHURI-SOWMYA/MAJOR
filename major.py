import streamlit as st
import numpy as np
from PIL import Image

# Function to process the uploaded image
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the processing size (224x224)
    return np.array(img)

# Function to compute the Mean Squared Error (MSE) between two patches
def mse(patchA, patchB):
    # Compute the Mean Squared Error between two patches
    err = np.sum((patchA - patchB) ** 2)
    err /= float(patchA.shape[0] * patchA.shape[1])
    return err

# Function to check for copy-move forgery based on pixel comparison
def detect_copy_move_forgery(img_array):
    height, width, _ = img_array.shape
    patch_size = 64  # Patch size for comparison
    threshold = 1000  # Threshold for MSE, adjust based on sensitivity
    
    # Compare each patch with every other patch using MSE
    for i in range(0, height - patch_size, patch_size):
        for j in range(0, width - patch_size, patch_size):
            patch = img_array[i:i + patch_size, j:j + patch_size]
            
            for x in range(0, height - patch_size, patch_size):
                for y in range(0, width - patch_size, patch_size):
                    if i == x and j == y:
                        continue  # Skip comparing the patch with itself
                    
                    other_patch = img_array[x:x + patch_size, y:y + patch_size]
                    
                    # Compute the MSE between the two patches
                    error = mse(patch, other_patch)
                    if error < threshold:  # Threshold for detecting similar patches
                        return True  # Match found, indicating possible forgery
    return False  # No matches found, no forgery detected

# Streamlit App UI
st.title("Copy-Move Forgery Detection Using MSE")
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
