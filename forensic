import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# Function to process the uploaded image
def process_image(uploaded_image):
    img = Image.open(uploaded_image)
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((224, 224))  # Resize to fit the processing size (224x224)
    return np.array(img), img

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
    forged_regions = []

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
                        forged_regions.append(((i, j), (x, y)))  # Store locations of forged patches
    return forged_regions

# Function to mark the forged areas on the image
def mark_forgery_area(img, forged_regions):
    # Ensure the image is in RGB mode
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    
    for (start, end) in forged_regions:
        x1, y1 = start
        x2, y2 = end
        # Ensure the coordinates are within bounds of the image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, img.width)
        y2 = min(y2, img.height)

        # Draw rectangles around the forged regions
        try:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        except Exception as e:
            print(f"Error drawing rectangle: {e}")
    
    return img

# Streamlit App UI
st.title("Copy-Move Forgery Detection with Marked Forgery Areas")
st.write("Upload an image to detect copy-move forgery and see the forgery marked.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, img = process_image(uploaded_file)

    # Image details
    image_type = img.format
    image_size = img.size  # (width, height)
    
    st.write(f"**Image Type:** {image_type}")
    st.write(f"**Image Dimensions:** {image_size[0]} x {image_size[1]} pixels")

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    st.write("")
    st.write("Classifying... Please wait...")

    # Get prediction from the forgery detection function
    forged_regions = detect_copy_move_forgery(img_array)

    if forged_regions:
        st.write("Forgery Detected! The following areas are marked in the image.")
        
        # Mark the forged areas on the image
        marked_image = mark_forgery_area(img.copy(), forged_regions)

        # Display the marked image
        st.image(marked_image, caption="Image with Marked Forged Areas", use_column_width=True)
    else:
        st.write("No Forgery Detected!")
