import cv2
from svmutil import *


def draw_detections(img, rect, thickness=1):
    for x, y, w, h in rect:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

detect_path = './test_performance/'

img = cv2.imread(detect_path+'01.avi-4.jpg')
win_size = (40, 40)
block_size = (20, 20)
block_stride = (10, 10)
cell_size = (10, 10)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)


#hog.setSVMDetector('data.model')
model = svm_load_model('data.model')


found, w = hog.detectMultiScale(img, winStride=(10, 10), scale=1.05)


draw_detections(img, found)
cv2.imshow('test', img)
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
