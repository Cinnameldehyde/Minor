import streamlit as st
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
import os

# Load labels from labels.txt
def load_labels(filename):
    labels_dict = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    key, value = parts
                    labels_dict[int(key)] = value.strip()
    except FileNotFoundError:
        st.error("Error: labels.txt file not found!")
    except ValueError:
        st.error("Error: Invalid format in labels.txt!")
    return labels_dict

# Check if model file exists
MODEL_PATH = "model3.tflite"
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: {MODEL_PATH} not found!")
else:
    # Load TFLite model
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load label names
    labels = load_labels("labels.txt")

    # CSS for animated floating blocks with hover pause and clickable effect
    st.markdown("""
        <style>
            body {
                background-color: #f5f5f5;
            }

            @keyframes marqueeAnimation {
                from { transform: translateX(50%); }
                to { transform: translateX(-50%); }
            }

            .marquee-container {
                display: flex;
                justify-content: center;
                overflow: hidden;
                width: 100%;
                padding: 25px 35px;
            }

            .marquee {
                display: flex;
                gap: 10px 25px;
                animation: marqueeAnimation 30s linear infinite;
            }

            .marquee div {
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(70deg, #61413a, #3a6156);
                padding: 10px 10px;
                border-radius: 15px;
                font-size: 25px;
                font-weight: bold;
                text-align: center;
                min-width: 180px;
                max-width: 250px;
                max-height: 250px;
                white-space: wrap;
                box-shadow: 5px 5px 5px rgba(255, 255, 255, 0.3);
                transition: transform 0.3s ease-in-out;
                cursor: pointer;
            }

            .marquee:hover {
                animation-play-state: paused;
            }

            .marquee div:hover {
                transform: scale(1.1);
            }
        </style>
    """, unsafe_allow_html=True)

    # Floating Blocks Display
    if labels:
        st.markdown("""
            <div class="marquee-container">
                <div class="marquee">
        """ + "".join(f'<div>{label}</div>' for label in labels.values()) + """
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Streamlit UI
    st.title("🚦 Road Sign Recognition Web App")
    st.write("Upload an image of a road sign to identify it.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a road sign image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        image = image.resize((32, 32))  # Resize to match model input
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=(0, -1))  # Ensure correct shape (1, 32, 32, 1)

        # Ensure correct dtype
        if image.dtype != input_details[0]['dtype']:
            image = image.astype(input_details[0]['dtype'])

        # Normalize if required
        if np.max(image) > 1:
            image /= 255.0

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])
        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data) * 100  # Convert confidence to percentage

        with col2:
            st.subheader("Prediction:")
            detected_label = labels.get(predicted_class, "Unknown Sign")
            st.markdown(f"<h3 style='color: yellow;'>{detected_label}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px; color: cyan;'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
