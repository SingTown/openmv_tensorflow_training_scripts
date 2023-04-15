# This code run in OpenMV4 H7 Plus

import sensor, image, time, os, tf

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

clock = time.clock()
while(True):
    clock.tick()
    img = sensor.snapshot()
    for obj in tf.classify("trained.tflite", img, min_scale=1.0, scale_mul=0.5, x_overlap=0.0, y_overlap=0.0):
        img.draw_rectangle(obj.rect())
        print(obj.output())
    print(clock.fps(), "fps")
