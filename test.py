import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Function to preprocess the image
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return img

# Load the trained model
MODEL_PATH = "_trained-models/rock-paper-scissors"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ['paper', 'rock', 'scissors']  # Adjust according to your dataset

# Test images directory
TEST_IMAGES_DIRECTORY = Path("rock-paper-scissors/test-dataset")

# Get list of test image paths
test_image_paths = list(TEST_IMAGES_DIRECTORY.glob("*.jpg"))

# Display 5 images with their actual and predicted classes
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, ax in enumerate(axes):
    # Preprocess the image
    image_path = test_image_paths[i]
    image = preprocess_image(image_path)
    ax.imshow(image)
    ax.axis('off')

    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    actual_class = ""
    if image_path.name.startswith("paper"):
        actual_class = "paper"
    elif image_path.name.startswith("rock"):
        actual_class = "rock"
    elif image_path.name.startswith("scissors"):
        actual_class = "scissors"

    # Display result
    ax.set_title(f"Actual: {actual_class}, Predicted: {predicted_class}")

plt.tight_layout()
plt.show()
