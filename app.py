import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import pandas as pd
import io
import requests
import cv2
import os
import base64
import json


def convert_image(image, output_format):
    image = Image.open(image)
    output = io.BytesIO()
    image.convert('RGB').save(output, format=output_format)
    return output


def classify_image(image):
    # Load a pre-trained model
    model = MobileNetV2(weights='imagenet')

    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Make predictions
    predictions = model.predict(image_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    return [(label, confidence) for _, label, confidence in decoded_predictions]


def call_huggingface_api(image, api_url, headers):
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Create the payload
    payload = {
        "inputs": img_str
    }

    response = requests.post(api_url, headers=headers, json=payload)
    try:
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.HTTPError as e:
        if response.status_code == 413:  # Payload too large
            return {"error": "Payload Too Large"}
        else:
            st.error(f"HTTP error occurred: {e}")
            return {"error": "HTTP error occurred"}

    try:
        return response.json()
    except json.JSONDecodeError:
        st.error(f"Failed to decode JSON response: {response.text}")
        return {"error": "Failed to decode JSON response"}


def compress_image(image, max_size=(800, 800)):
    # Compress the image by resizing it using LANCZOS filter
    image.thumbnail(max_size, Image.LANCZOS)
    return image


def draw_detections(image, detections):
    for detection in detections:
        if 'label' in detection and 'box' in detection:
            bbox = detection['box']
            label = detection['label']
            score = detection['score']
            x, y, w, h = bbox['xmin'], bbox['ymin'], bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


st.title("Image Processing and AI Features Web App")

# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Convert Images", "Classify Images", "Object Detection"],
        icons=["image", "list-task", "camera"],
        menu_icon="cast",
        default_index=0,
    )

# Convert Images Page
if selected == "Convert Images":
    st.header("Convert Images")
    uploaded_file = st.file_uploader("Upload an image (WEBP, JPG, JPEG, PNG)", type=['webp', 'jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        with st.expander("Uploaded Image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Identify image format
        format_mapping = {
            "JPEG": "jpg",
            "PNG": "png",
            "WEBP": "webp"
        }
        image_format = image.format
        st.write(f"Uploaded image format: {image_format}")

        # Image conversion feature
        output_format_options = [fmt for fmt in format_mapping if fmt != image_format]
        output_format = st.selectbox("Select output format", output_format_options)
        if st.button("Convert"):
            output = convert_image(uploaded_file, output_format)
            st.download_button(
                label="Download converted image",
                data=output,
                file_name=f"converted.{format_mapping[output_format]}",
                mime=f"image/{format_mapping[output_format]}"
            )

# Classify Images Page
elif selected == "Classify Images":
    st.header("Classify Images")
    uploaded_file = st.file_uploader("Upload an image for classification (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        with st.expander("Uploaded Image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image for Classification', use_column_width=True)

        if st.button("Classify"):
            with st.spinner("Classifying..."):
                predictions = classify_image(image)

            # Display predictions in a table
            df = pd.DataFrame(predictions, columns=['Label', 'Confidence'])
            df['Confidence'] = df['Confidence'].apply(lambda x: f"{x * 100:.2f}%")
            with st.expander("Predicted Classes"):
                st.table(df)

# Object Detection Page
elif selected == "Object Detection":
    st.header("Object Detection")
    uploaded_file = st.file_uploader("Upload an image for object detection (JPG, JPEG, PNG)",
                                     type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        with st.expander("Uploaded Image"):
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image for Object Detection', use_column_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                API_URL_objDet = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"
                headers_objDet = {"Authorization": "Bearer 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"}

                detections = call_huggingface_api(image, API_URL_objDet, headers_objDet)

                if isinstance(detections, dict) and detections.get("error") == "Payload Too Large":
                    st.warning("Image is too large, compressing and trying again.")
                    with st.spinner("Compressing image..."):
                        image = compress_image(image)
                    detections = call_huggingface_api(image, API_URL_objDet, headers_objDet)

                if isinstance(detections, dict) and "error" in detections:
                    st.error(detections["error"])
                elif isinstance(detections, list):
                    # Extract relevant information for table display
                    detection_info = [{"Label": det["label"], "Score": f"{det['score'] * 100:.2f}%"} for det in
                                      detections]
                    df = pd.DataFrame(detection_info)
                    st.write("Detected Objects:")
                    with st.expander("Object Detection Results"):
                        st.table(df)

                    # Draw detections on the image
                    image_np = np.array(image)
                    detected_image = draw_detections(image_np, detections)
                    st.image(detected_image, caption='Detected Objects', use_column_width=True)
                else:
                    st.error("Unexpected response format.")
