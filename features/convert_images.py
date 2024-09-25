import streamlit as st
from utils.image_utils import convert_image
from PIL import Image

def convert_images_page():
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
