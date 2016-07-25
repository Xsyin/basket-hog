import cv2
import os
import random

path = '/home/xsyin/videos/'
# rectangle formate is (x, y, w, h)
rects = {'01.avi': (883, 79, 80, 80), '01.mov': (505, 270, 48, 48), '02.avi': (951, 219, 74, 74), '02.mov': (475, 291, 42, 42), '03.avi': (
    664, 90, 71, 71), '03.mov': (472, 292, 40, 40), '04.avi': (1196, 246, 72, 72), '04.mov': (473, 291, 43, 43)}
# (w,h) windows size of cut box
cut_size = (40, 40)
# step is the gap between tow frames
step = 1000
for file in os.listdir(path):
    video_path = path + file
    file = file.lower()
    if file not in rects:
        continue
    print "Now is cutting image from " + file
    rect = rects[file]
    capture = cv2.VideoCapture(video_path)
    frame_cout = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    frame_width = capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    # the position of frame in video
    pos = 25
    while capture.isOpened():
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        ret, frame = capture.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xgap = (rect[0] - cut_size[0], rect[0] + rect[2])
        ygap = (rect[1] - cut_size[1], rect[1] + rect[3])

        for j in xrange(10):
            x = random.randint(0, frame_width - cut_size[0])
            y = random.randint(0, frame_height - cut_size[1])
            while xgap[0] < x < xgap[1] and ygap[0] < y < ygap[1]:
                x = random.randint(0, frame_width)
                y = random.randint(0, frame_height)
            image = frame[y:y + cut_size[1], x:x + cut_size[0]]
            cv2.imwrite(('negative/%s-%d-%d.jpg') %
                        (file, pos / step, j), image)

        if pos <= frame_cout - step:
            pos += step
        else:
            break

print 'end!'
capture.release()
