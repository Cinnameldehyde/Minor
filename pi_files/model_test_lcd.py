import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import smbus
import time

# Define some device parameters
I2C_ADDR = 0x27  # I2C device address
LCD_WIDTH = 16  # Maximum characters per line

# Define some device constants
LCD_CHR = 1  # Mode - Sending data
LCD_CMD = 0  # Mode - Sending command

LCD_LINE_1 = 0x80  # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0  # LCD RAM address for the 2nd line
LCD_LINE_3 = 0x94  # LCD RAM address for the 3rd line
LCD_LINE_4 = 0xD4  # LCD RAM address for the 4th line

LCD_BACKLIGHT = 0x08  # On
ENABLE = 0b00000100  # Enable bit

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005

# Open I2C interface
bus = smbus.SMBus(1)

# Initialize the display
def lcd_init():
    lcd_byte(0x33, LCD_CMD)  # 110011 Initialise
    lcd_byte(0x32, LCD_CMD)  # 110010 Initialise
    lcd_byte(0x06, LCD_CMD)  # 000110 Cursor move direction
    lcd_byte(0x0C, LCD_CMD)  # 001100 Display On,Cursor Off, Blink Off
    lcd_byte(0x28, LCD_CMD)  # 101000 Data length, number of lines, font size
    lcd_byte(0x01, LCD_CMD)  # 000001 Clear display
    time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
    bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT

    # High bits
    bus.write_byte(I2C_ADDR, bits_high)
    lcd_toggle_enable(bits_high)

    # Low bits
    bus.write_byte(I2C_ADDR, bits_low)
    lcd_toggle_enable(bits_low)

def lcd_toggle_enable(bits):
    time.sleep(E_DELAY)
    bus.write_byte(I2C_ADDR, (bits | ENABLE))
    time.sleep(E_PULSE)
    bus.write_byte(I2C_ADDR, (bits & ~ENABLE))
    time.sleep(E_DELAY)

def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH, " ")
    lcd_byte(line, LCD_CMD)
    for i in range(LCD_WIDTH):
        lcd_byte(ord(message[i]), LCD_CHR)

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
input_shape = input_details[0]['shape']

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[2], input_shape[1]))
    if input_shape[3] == 1:  # Grayscale input
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
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

# Set lower resolution and frame rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32)
cap.set(cv2.CAP_PROP_FPS, 15)

print("Press 'q' to quit.")

frame_skip = 3  # Only process every 3rd frame
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to save processing time

    # Preprocess the frame
    input_data = preprocess_image(frame)

    # Set the tensor input
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Extract the predicted class and confidence
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)

    # Display results if confidence is above a threshold
    if confidence > 0.6:  # Confidence threshold
        label = classes.get(predicted_class, "Unknown")
        try:
            lcd_init()
            lcd_string(label, LCD_LINE_1)
            lcd_string(f"{confidence:.2f}", LCD_LINE_2)  # Format confidence to 2 decimal places
        except KeyboardInterrupt:
            pass
        finally:
            lcd_byte(0x01, LCD_CMD)  # Clear LCD after showing result

    # Show the frame with classification result
    cv2.imshow("Road Sign Classification", cv2.resize(frame, (32, 32)))

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
