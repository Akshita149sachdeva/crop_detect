import streamlit as st # type: ignore
import torch # type: ignore
import cv2
import numpy as np
from PIL import Image # type: ignore
from ultralytics import YOLO # type: ignore

# Load the YOLO model
model_path = "runs/detect/train10/weights/best.pt"
model = YOLO(model_path)

# Streamlit UI
st.title("Crop Detection using YOLO")
st.write("Upload an image to detect crops.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Run YOLO model
    results = model(image_np)
    
    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("### Detection Results:")
    
    for result in results:
        st.image(result.plot(), caption="Detected Crops", use_column_width=True)
