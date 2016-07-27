import cv2
import numpy as np
import xml.etree.ElementTree as ET


def read_config():
    configs = dict()
    fp = open('config.txt', 'r')
    str_config = fp.readlines()
    for line in str_config:
        line = line.strip('')
        if line.find('pos_path') >= 0:
            pos_path = line.split('=')[1].strip()
            configs['pos_path'] = pos_path
        if line.find('neg_path') >= 0:
            neg_path = line.split('=')[1].strip()
            configs['neg_path'] = neg_path

    return configs


def gen_file_for_grid(i, data_mat, label_mat):
    fp = open('/home/xsyin/libsvm-3.21/tools/data.txt', 'w')
    for j in xrange(0, i-1):
        if label_mat[j] == 1:
            s = '+1 '
        else:
            s = '-1 '
        for k in xrange(0, len(data_mat[j])):
            s = s+str(k+1)+':'+str(data_mat[j][k])+' '
        fp.write(s+'\n')
    fp.close()


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
