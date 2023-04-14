import Jetson.GPIO as GPIO
import time
import signal
import sys


GPIO.setmode(GPIO.BOARD)

enA = 33
in1 = 7
in2 = 11
# thumb_abd_pin = 0
# thumb_add_pin = 0

GPIO.setup(enA, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.output(in1, GPIO.HIGH)
GPIO.output(in2, GPIO.LOW)

index = GPIO.PWM(enA, 50)

index.start(0)
# pw = 1 + (100/180) * (2-1)
# dc = pw/(1/50)*100
dc = 20
index.ChangeDutyCycle(dc)


time.sleep(5)

# # Pin Definition
# led_pin = 33
 
# # Set up the GPIO channel
# GPIO.setmode(GPIO.BOARD) 
# GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH) 
 
# print("Press CTRL+C when you want the LED to stop blinking") 
 
# # Blink the LED
# while True: 
#   time.sleep(2) 
#   GPIO.output(led_pin, GPIO.HIGH) 
#   print("LED is ON")
#   time.sleep(2) 
#   GPIO.output(led_pin, GPIO.LOW)
#   print("LED is OFF")

# btn_pin = 7
# GPIO.setup(btn_pin, GPIO.IN)

# while True:
#     try:
#       time.sleep(0.25)
#       print(GPIO.input(btn_pin))
#     except KeyboardInterrupt:
#       print("")
#       print(":(")
#       GPIO.cleanup()
#       sys.exit()
#       break

GPIO.cleanup()

print('Done.')
