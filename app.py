import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
import streamlit as st
from PIL import Image

# Load the TensorFlow Lite model
interpreter = Interpreter(model_path="model3.tflite")
interpreter.allocate_tensors()

# Get input and output details from the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the labels from labels.txt
def load_labels():
    with open("labels.txt", "r") as f:
        labels = f.readlines()
    # Strip newline characters from each label
    labels = [label.strip() for label in labels]
    return labels

# Define a function to preprocess the image (you can adjust this based on your model's requirements)
def preprocess_image(image):
    # Resize the image to the model's expected input size (32x32 for your model)
    image = image.resize((32, 32))  # Adjusted to 32x32

    # Convert image to grayscale if the model expects a single channel input
    image = image.convert('L')  # Convert to grayscale
    
    # Convert image to numpy array and normalize (if required by your model)
    image = np.array(image)

    # Normalize to [0, 1] range (if your model was trained this way)
    image = image.astype(np.float32)
    image /= 255.0

    # Add batch dimension (1, height, width, channels) with 1 channel
    image = np.expand_dims(image, axis=-1)  # Grayscale image, so 1 channel

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# Define the function to make predictions
def predict(image):
    # Preprocess the image to match model input
    input_data = preprocess_image(image)
    
    # Set the tensor with the processed image data
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))  # Convert to FLOAT32

    # Run inference
    interpreter.invoke()

    # Get the output data (prediction results)
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the index of the class with the highest probability (softmax output)
    predicted_class_index = np.argmax(output_data)

    # Load the labels
    labels = load_labels()

    # Get the predicted label
    predicted_label = labels[predicted_class_index]

    # Get the confidence score (the probability for the predicted class)
    confidence_score = output_data[0][predicted_class_index]

    return predicted_label, confidence_score

# Streamlit Web Interface for uploading image
st.title("TensorFlow Lite Model Inference for Road Signs")
st.write("Upload an image to make a prediction")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)

    # Display the image on the web app
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    label, confidence = predict(image)

    # Display the prediction result with confidence score
    st.write(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence * 100:.2f}%")  # Display as percentage
