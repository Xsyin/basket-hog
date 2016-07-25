import cv2
import numpy as np
import os
from svmutil import *

pos_path = './positive/'
neg_path = './negative/'

pos_num = len(os.listdir(pos_path))
print 'positive set num is '+str(pos_num)
neg_num = len(os.listdir(neg_path))
print 'negative set num is '+str(neg_num)
total_num = pos_num + neg_num


win_size = (40, 40)
block_size = (20, 20)
block_stride = (10, 10)
cell_size = (10, 10)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

wins = ((win_size[0] - block_size[0]) / block_stride[0] + 1) * \
       ((win_size[1] - block_size[1]) / block_stride[1] + 1)
cells = (block_size[0] / cell_size[0]) * (block_size[1] / cell_size[1])
hog_dimension = wins * cells * nbins


data_mat = np.empty((total_num, hog_dimension), dtype=np.float32)
pos_label = np.ones((pos_num, 1), dtype=np.float32)
neg_label = np.zeros((neg_num, 1), dtype=np.float32)
label_mat = np.vstack((pos_label, neg_label))


pos_img_path = [pos_path + x for x in os.listdir(pos_path)]
neg_img_path = [neg_path + x for x in os.listdir(neg_path)]
img_path = pos_img_path + neg_img_path
i = 0
for image in img_path:
    img = cv2.imread(image)
    desc = hog.compute(img, block_stride)
    data_mat[i] = desc.T
    i += 1


#generate data.txt for libsvm format

# fp = open('./data.txt', 'w')
# for j in xrange(0, i-1):
#     if label_mat[j] == 1:
#         s = '+1 '
#     else:
#         s = '-1 '
#     for k in xrange(0, len(data_mat[j])):
#         s = s+str(k+1)+':'+str(data_mat[j][k])+' '
#     fp.write(s+'\n')
# fp.close()

y, x = svm_read_problem('data.txt')
m = svm_train(y[:5000], x[:5000], '-c 8')
svm_save_model('data.model', m)
p_label, p_acc, p_val = svm_predict(y, x, m)

ACC, MSE, SCC = evaluations(y, p_label)

















