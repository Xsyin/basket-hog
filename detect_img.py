import cv2
from util import draw_detections, get_svm_vector
import os


if __name__ == '__main__':
    detect_path = './test_performance/'

    win_size = (40, 40)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    vect = get_svm_vector('data_pos_3496_neg_11777.xml')
    hog.setSVMDetector(vect)
    img_name = '03.MOV-3.jpg'
    #for img_name in os.listdir(detect_path):
    img = cv2.imread(detect_path+img_name)

    if img_name.find('avi') >= 0:
        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))

    found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)
    draw_detections(img, found)
    cv2.imshow('test', img)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
