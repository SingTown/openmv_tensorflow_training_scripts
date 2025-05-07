# This code run in OpenMV H7, OpenMV H7 Plus and OpenMV RT

import sensor
import time
import ml

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.GRAYSCALE)  # Set pixel format to GRAYSCALE (or RGB565)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))  # Set 240x240 window.
sensor.skip_frames(time=2000)  # Let the camera adjust.

model = ml.Model("trained.tflite", load_to_fb=True)
norm = ml.Normalization(scale=(0, 1.0))

clock = time.clock()
while True:
    clock.tick()

    img = sensor.snapshot().binary([(0,60)]).dilate(2)
    input = [norm(img)] # scale 0~255 to 0~1.0
    result = model.predict(input)[0].flatten().tolist()
    number = result.index(max(result))
    print("number", number)
    print(clock.fps(), "fps")
