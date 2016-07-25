import numpy as np
import cv2
from detect_img import get_svm_vector

path = '/home/xsyin/videos/'

def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    win_size = (40, 40)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    video_path = path + '03.avi'
    vector = get_svm_vector('svm_data.xml')

    hog.setSVMDetector(vector)
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        found, w = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.1)
        draw_detections(frame, found)
        cv2.imshow('feed', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cap.release()
cv2.destroyAllWindows()
