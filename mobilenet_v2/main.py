# This code run in OpenMV H7 Plus and OpenMV RT

import sensor
import time
import ml

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))  # Set 240x240 window.
sensor.skip_frames(time=2000)  # Let the camera adjust.

model = ml.Model("trained.tflite", load_to_fb=True)

clock = time.clock()
while True:
    clock.tick()

    img = sensor.snapshot()
    result = model.predict([img])[0].flatten().tolist()
    print("cat:", result[0], "dog", result[1])
    print(clock.fps(), "fps")
