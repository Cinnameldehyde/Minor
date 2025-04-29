import RPi.GPIO as GPIO
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import smbus
import time

# ------------------ GPIO Motor Setup ------------------
in1, in2, en_a = 17, 27, 4
in3, in4, en_b = 5, 6, 13

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

for pin in [in1, in2, en_a, in3, in4, en_b]:
    GPIO.setup(pin, GPIO.OUT)

pwm_a = GPIO.PWM(en_a, 100)
pwm_b = GPIO.PWM(en_b, 100)
pwm_a.start(0)
pwm_b.start(0)

# ------------------ LCD Setup ------------------
I2C_ADDR, LCD_WIDTH = 0x27, 16
LCD_CHR, LCD_CMD = 1, 0
LCD_LINE_1, LCD_LINE_2 = 0x80, 0xC0
LCD_BACKLIGHT, ENABLE = 0x08, 0b00000100
E_PULSE, E_DELAY = 0.0005, 0.0005
bus = smbus.SMBus(1)

def lcd_init():
    for cmd in [0x33, 0x32, 0x06, 0x0C, 0x28, 0x01]:
        lcd_byte(cmd, LCD_CMD)
        time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    bus.write_byte(I2C_ADDR, mode | (bits & 0xF0) | LCD_BACKLIGHT)
    lcd_toggle_enable(mode | (bits & 0xF0) | LCD_BACKLIGHT)
    bus.write_byte(I2C_ADDR, mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT)
    lcd_toggle_enable(mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT)

def lcd_toggle_enable(bits):
    time.sleep(E_DELAY)
    bus.write_byte(I2C_ADDR, (bits | ENABLE))
    time.sleep(E_PULSE)
    bus.write_byte(I2C_ADDR, (bits & ~ENABLE))
    time.sleep(E_DELAY)

def lcd_string(message, line):
    message = message.ljust(LCD_WIDTH)
    lcd_byte(line, LCD_CMD)
    for char in message:
        lcd_byte(ord(char), LCD_CHR)

# ------------------ TFLite Model Setup ------------------
interpreter = tflite.Interpreter(model_path="model3.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

classes = {
    2: 'Speed Limit 50 km/h',
    5: 'Speed Limit 80 km/h',
    14: 'Stop'
}

def preprocess_image(image):
    image = cv2.resize(image, (input_shape[2], input_shape[1]))
    if input_shape[3] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    image = image / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)

# ------------------ Motor Control ------------------
def move_forward(speed_percent):
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.HIGH)
    GPIO.output(in4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed_percent)
    pwm_b.ChangeDutyCycle(speed_percent)

def stop_motors():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

# ------------------ Main ------------------
lcd_init()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 32)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 32)
cap.set(cv2.CAP_PROP_FPS, 10)

frame_skip = 5
frame_count = 0
previous_label = ""
current_speed = -1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera Error!")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        input_data = preprocess_image(frame)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        confidence = np.max(output_data)

        if confidence > 0.7:
            label = classes.get(predicted_class, "Unknown")
            if label != previous_label:
                lcd_init()
                lcd_string(label, LCD_LINE_1)
                lcd_string(f"{confidence:.2f}", LCD_LINE_2)
                previous_label = label

                if label == 'Speed Limit 50 km/h' and current_speed != 50:
                    move_forward(50)
                    current_speed = 50
                elif label == 'Speed Limit 80 km/h' and current_speed != 100:
                    move_forward(100)
                    current_speed = 100
                elif label == 'Stop':
                    stop_motors()
                    current_speed = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
