import cv2
import numpy as np
import xml.etree.ElementTree as ET


def draw_detections(img, rects, thickness=2):
    for x, y, w, h in rects:
        #pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        #cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness)


def get_svm_vector(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    sv_elem = root.find('my_svm').find('support_vectors')
    var_count = int(root.find('my_svm').find('var_count').text)
    sv_total = int(root.find('my_svm').find('sv_total').text)

    svs = np.empty((sv_total, var_count), dtype=np.float32)
    i = 0
    while i < sv_total:
        aa = sv_elem[i].text
        svs[i] = np.array(aa.split(), dtype=np.float32)
        i += 1
    svs = np.matrix(svs)

    alpha = root.find('my_svm').find('decision_functions')[
        0].find('alpha').text.split()
    alpha = np.matrix(alpha, dtype=np.float32)

    rho = root.find('my_svm').find('decision_functions')[0].find('rho')
    rho = float(rho.text)

    vector = -alpha * svs
    vector = np.c_[vector, rho]
    return vector.T


if __name__ == '__main__':
    detect_path = './test_performance/'
    img_name = '03.MOV-3.jpg'
    img = cv2.imread(detect_path+img_name)

    win_size = (40, 40)
    block_size = (10, 10)
    block_stride = (5, 5)
    cell_size = (5, 5)

    # block_size = (20, 20)
    # block_stride = (10, 10)
    # cell_size = (10, 10)

    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

    vect = get_svm_vector('data_pos_3014_neg_3050.xml')

    if img_name.find('avi') >= 0:
        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
    hog.setSVMDetector(vect)
    found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.1)

    draw_detections(img, found)
    cv2.imshow('test', img)
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows()
