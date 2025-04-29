import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Right Motor
in1 = 17
in2 = 27
en_a = 4

# Left Motor
in3 = 5
in4 = 6
en_b = 13

# Setup pins
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(en_a, GPIO.OUT)

GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(en_b, GPIO.OUT)

# Setup PWM
q = GPIO.PWM(en_a, 100)  # Right motor PWM
p = GPIO.PWM(en_b, 100)  # Left motor PWM

q.start(0)
p.start(0)

# Set direction forward
GPIO.output(in1, GPIO.HIGH)
GPIO.output(in2, GPIO.LOW)

GPIO.output(in3, GPIO.HIGH)
GPIO.output(in4, GPIO.LOW)

try:
    speeds = [25, 50, 75, 100]  # List of speeds (duty cycles)

    while True:
        for speed in speeds:
            print(f"Running at {speed}% speed")
            q.ChangeDutyCycle(speed)
            p.ChangeDutyCycle(speed)
            sleep(2)  # Run at this speed for 2 seconds

        # After full cycle, stop for 2 seconds
        print("Stopping...")
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)
        q.ChangeDutyCycle(0)
        p.ChangeDutyCycle(0)
        sleep(2)

except KeyboardInterrupt:
    print("Interrupted. Cleaning up...")
    GPIO.cleanup()
