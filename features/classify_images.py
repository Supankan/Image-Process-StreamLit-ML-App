import streamlit as st
from PIL import Image
from utils.model_utils import classify_image
import pandas as pd


def classify_images_page():
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
