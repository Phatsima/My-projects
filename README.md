# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained model (EfficientNetB0 trained on ImageNet)
model = EfficientNetB0(weights='imagenet')

def classify_animal_species(img_path, top_k=3):
    """
    Classify animal species in an image using a pre-trained model.
    
    Args:
        img_path (str): Path to the input image.
        top_k (int): Number of top predictions to return.
    
    Returns:
        List of predicted species and their confidence scores.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=top_k)[0]

    # Extract results
    results = []
    for _, species, confidence in decoded_predictions:
        results.append({"species": species, "confidence": float(confidence)})

    return results

# Example usage
if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"  # Replace with your image path
    predictions = classify_animal_species(img_path)

    # Display the image and predictions
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    print("\nTop Predictions:")
    for pred in predictions:
        print(f"- {pred['species']}: {pred['confidence']*100:.2f}%")