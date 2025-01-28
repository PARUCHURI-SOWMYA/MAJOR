import streamlit as st
from PIL import Image
import numpy as np
import imagehash

# Function to process the uploaded image
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the processing size (224x224)
    return img

# Function to compute hash for an image
def get_image_hash(img):
    return imagehash.phash(img)  # Perceptual hash to detect similar regions

# Function to check for copy-move forgery based on image hashing
def detect_copy_move_forgery(img):
    # Create hash for the original image
    original_hash = get_image_hash(img)
    
    # Create small patches of the image and compare hashes
    patch_size = 64  # Patch size for comparison
    img_array = np.array(img)
    height, width, _ = img_array.shape

    # Scan the image by patches
    for i in range(0, height - patch_size, patch_size):
        for j in range(0, width - patch_size, patch_size):
            patch = img_array[i:i + patch_size, j:j + patch_size]
            patch_img = Image.fromarray(patch)
            patch_hash = get_image_hash(patch_img)
            
            # Compare hashes for similarity
            if original_hash - patch_hash < 5:  # Adjust the threshold as needed
                return True  # Forgery detected
    return False  # No forgery detected

# Streamlit App UI
st.title("Copy-Move Forgery Detection Using Image Hashing")
st.write("Upload an image to detect copy-move forgery.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = process_image(uploaded_file)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    st.write("Classifying... Please wait...")

    # Get prediction from the forgery detection function
    is_forged = detect_copy_move_forgery(img)

    if is_forged:
        st.write("Forgery Detected!")
    else:
        st.write("No Forgery Detected!")
