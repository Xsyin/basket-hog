import cv2
import numpy as np
import os

pos_path = './positive/'
neg_path = './negative/'

pos_num = len(os.listdir(pos_path))
print 'positive set num is '+str(pos_num)
neg_num = len(os.listdir(neg_path))
print 'negative set num is '+str(neg_num)
total_num = pos_num + neg_num


win_size = (40, 40)
block_size = (10, 10)
block_stride = (5, 5)
cell_size = (5, 5)
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

#generate data.txt for grid.py to get best SVM params
#
# print 'write begin !'
# fp = open('/home/xsyin/libsvm-3.21/tools/data.txt', 'w')
# for j in xrange(0, i-1):
#     if label_mat[j] == 1:
#         s = '+1 '
#     else:
#         s = '-1 '
#     for k in xrange(0, len(data_mat[j])):
#         s = s+str(k+1)+':'+str(data_mat[j][k])+' '
#     fp.write(s+'\n')
# fp.close()
# print 'write end !'

svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                  svm_type=cv2.SVM_C_SVC)
print 'begin train SVM...'
svm = cv2.SVM()
svm.train_auto(data_mat, label_mat, varIdx=None, sampleIdx=None, params=svm_params)
svm.save('svm_data2.xml')
print 'svm_data2.xml have saved !'
















