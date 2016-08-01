import cv2
from util import draw_detections, get_svm_vector


path = '/home/xsyin/video/'

if __name__ == '__main__':

    win_size = (40, 40)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    video_path = path + '02.mp4'
    vector = get_svm_vector('svm_data.xml')
    hog.setSVMDetector(vector)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        _, frame = cap.read()
        if i%100 == 0:
            found, w = hog.detectMultiScale(frame, winStride=(8, 8), scale=1.02)
        draw_detections(frame, found)
        cv2.imshow('video', frame)
        i += 1
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cap.release()
cv2.destroyAllWindows()
