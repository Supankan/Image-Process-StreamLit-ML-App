from PIL import Image
import io

def convert_image(image, output_format):
    image = Image.open(image)
    output = io.BytesIO()
    image.convert('RGB').save(output, format=output_format)
    return output
