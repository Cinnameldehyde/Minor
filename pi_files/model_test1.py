import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="model3.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Full class names dictionary
classes = {
    0: 'Speed Limit 20 km/h',
    1: 'Speed Limit 30 km/h',
    2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h',
    4: 'Speed Limit 70 km/h',
    5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h',
    7: 'Speed Limit 100 km/h',
    8: 'Speed Limit 120 km/h',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Input image dimensions
input_shape = input_details[0]['shape']  # Expected shape of the input tensor

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[2], input_shape[1]))  # Resize to the expected input shape
    if input_shape[3] == 1:  # If the model expects grayscale input
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Function to draw label and confidence
def draw_label(image, label, confidence, x, y):
    cv2.putText(image, f"{label} ({confidence*100:.1f}%)", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Open the video feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the entire frame (resize to model input size)
    input_data = preprocess_image(frame)

    # Set the tensor input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print the output data to inspect its structure
    # print("Output data:", output_data)

    # Extract the predicted class and confidence
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)

    # Display results if confidence is above a threshold
    if confidence > 0.6:  # Confidence threshold
        label = classes.get(predicted_class, "Unknown")
        h, w, _ = frame.shape
        draw_label(frame, label, confidence, 10, 30)  # Draw label and confidence

    # Show the frame with classification result
    cv2.imshow("Road Sign Classification", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()