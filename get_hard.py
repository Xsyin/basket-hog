# -*- coding: utf-8 -*-
import cv2
import os
from nms import non_max_suppression
from util import get_svm_vector

rects = {'01.avi': (884, 87, 80, 80), '01.mov': (505, 275, 48, 48), '02.avi': (951, 227, 74, 74), '02.mov': (475, 296, 42, 42), '03.avi': (
    664, 97, 71, 71), '03.mov': (472, 297, 40, 40), '04.avi': (1196, 251, 72, 72), '04.mov': (473, 296, 43, 43)}

win_size = (40, 40)
block_size = (10, 10)
block_stride = (5, 5)
cell_size = (5, 5)
nbins = 9
# 设置HoG描述子参数
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
hog.setSVMDetector(get_svm_vector('svm_data.xml'))

path = '/home/xsyin/videos/'
step = 200
count = 0
for file in os.listdir(path):
    file = file.lower()
    if file not in rects:
        continue
    print "Now is get hard example from " + file
    rect = rects[file]
    video_path = path + file
    file = file[0:2] + file[3:]
    capture = cv2.VideoCapture(video_path)
    frame_cout = capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    pos = 15
    while capture.isOpened():
        capture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        ret, frame = capture.read()
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objs, weights = hog.detectMultiScale(frame, scale=1.1)
        objs = non_max_suppression(objs)
        if len(objs) == 0:
            hoop = cv2.resize(
                frame[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]], (40, 40))
            cv2.imwrite('./pos_hard2/%s-%d.jpg' %
                        (file, pos / step), hoop)
        else:
            for (x, y, w, h) in objs:
                if rect[0] - 15 <= x <= rect[0] + 15 and rect[1] - 15 <= y <= rect[1] + 15:
                    continue
                nohoop = cv2.resize(
                    frame[y:y + h, x:x + w], (40, 40))
                count += 1
                cv2.imwrite('./neg_hard2/%s-%d.jpg' %
                            (file, count), nohoop)
        if pos <= frame_cout - step:
            pos += step
        else:
            break
capture.release()
print 'end!'
