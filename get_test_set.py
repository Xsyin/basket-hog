import cv2
import numpy as np
import os

path = '/home/xsyin/videos/'

for file in os.listdir(path):
    video_path = path + file
    print 'Start cut image from '+file
    capture = cv2.VideoCapture(video_path)
    step = 1000
    i = 0

    frame_count = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    while capture.isOpened():
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, i)
        ret, gray = capture.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('test_performance/'+file + '-'+str(i/step) + '.jpg', gray)
        if i <= frame_count - step:
            i += step
        else:
            break
print 'End!'
capture.release()

