import RPi.GPIO as GPIO
import smbus
import time

# -------- LCD CONFIG --------
I2C_ADDR = 0x27
LCD_WIDTH = 16

LCD_CHR = 1
LCD_CMD = 0

LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0

LCD_BACKLIGHT = 0x08
ENABLE = 0b00000100

E_PULSE = 0.0005
E_DELAY = 0.0005

bus = smbus.SMBus(1)

# -------- LDR & LED CONFIG --------
DO_PIN = 7  # LDR digital pin
LED_PIN = 8  # LED output pin

# -------- LCD FUNCTIONS --------
def lcd_init():
    lcd_byte(0x33, LCD_CMD)
    lcd_byte(0x32, LCD_CMD)
    lcd_byte(0x06, LCD_CMD)
    lcd_byte(0x0C, LCD_CMD)
    lcd_byte(0x28, LCD_CMD)
    lcd_byte(0x01, LCD_CMD)
    time.sleep(E_DELAY)

def lcd_byte(bits, mode):
    bits_high = mode | (bits & 0xF0) | LCD_BACKLIGHT
    bits_low = mode | ((bits << 4) & 0xF0) | LCD_BACKLIGHT
    bus.write_byte(I2C_ADDR, bits_high)
    lcd_toggle_enable(bits_high)
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

# -------- MAIN --------
def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(DO_PIN, GPIO.IN)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)

    lcd_init()

    last_state = None

    try:
        while True:
            light_state = GPIO.input(DO_PIN)

            if light_state == GPIO.LOW:
                status = "DAY"
                GPIO.output(LED_PIN, GPIO.LOW)  # turn OFF LED
            else:
                status = "NIGHT"
                GPIO.output(LED_PIN, GPIO.HIGH)  # turn ON LED

            if status != last_state:
                print(f"{status} mode")
                lcd_string("LDR Status:", LCD_LINE_1)
                lcd_string(status, LCD_LINE_2)
                last_state = status

            time.sleep(0.2)

    except KeyboardInterrupt:
        GPIO.cleanup()
        lcd_byte(0x01, LCD_CMD)
        print("\nProgram stopped.")

if __name__ == '__main__':
    main()
