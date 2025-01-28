import streamlit as st
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Function to process the uploaded image
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the processing size (224x224)
    return np.array(img)

# Function to compute a simple hash for a patch (sum of pixel values)
def compute_patch_hash(patch):
    return np.sum(patch)

# Function to check for copy-move forgery based on patch similarity
def detect_copy_move_forgery(img_array):
    height, width, _ = img_array.shape
    patch_size = 64  # Patch size for comparison
    
    # For simplicity, we use SSIM to compare patches (Structural Similarity Index)
    for i in range(0, height - patch_size, patch_size):
        for j in range(0, width - patch_size, patch_size):
            patch = img_array[i:i + patch_size, j:j + patch_size]
            
            # Compare this patch with the rest of the image using SSIM
            for x in range(0, height - patch_size, patch_size):
                for y in range(0, width - patch_size, patch_size):
                    if i == x and j == y:
                        continue  # Skip comparing the patch with itself
                    
                    other_patch = img_array[x:x + patch_size, y:y + patch_size]
                    
                    # Compute SSIM between the patches
                    similarity = ssim(patch, other_patch, multichannel=True)
                    if similarity > 0.9:  # Threshold for detecting similar patches
                        return True  # Match found, indicating possible forgery
    return False  # No matches found, no forgery detected

# Streamlit App UI
st.title("Copy-Move Forgery Detection Using SSIM")
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
