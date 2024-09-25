import requests
import io
import base64
import json
import cv2
from PIL import Image
import streamlit as st


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
