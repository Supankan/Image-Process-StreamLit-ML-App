import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

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
