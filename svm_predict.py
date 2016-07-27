import cv2
import os

test_path = './test/'
test_num = len(os.listdir(test_path))
print 'test set num is '+str(test_num)

win_size = (40, 40)
block_size = (10, 10)
block_stride = (5, 5)
cell_size = (5, 5)
# block_size = (20, 20)
# block_stride = (10, 10)
# cell_size = (10, 10)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

svm = cv2.SVM()
svm.load('svm_data.xml')


result = []
for image in os.listdir(test_path):
    test_img_path = test_path + image
    img = cv2.imread(test_img_path)
    desc = hog.compute(img, block_stride, (0, 0))
    res = svm.predict(desc)
    result.append(res)

print 'posivite num is '+str(result.count(1.0))
print 'negative num is '+str(result.count(0.0))
print 'posivite rate is '+str(float(result.count(1.0)) / test_num)
