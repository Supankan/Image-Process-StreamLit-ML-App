import streamlit as st
from PIL import Image
from utils.api_utils import call_huggingface_api, compress_image, draw_detections
import pandas as pd
import numpy as np

def object_detection_page():
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
