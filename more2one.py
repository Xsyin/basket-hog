import numpy as np
from PIL import Image
import os


img_mat = np.zeros((4, 2))

print img_mat
# path = './negative/'
# total_num = len(os.listdir(path))
# im_array = np.empty((total_num * 40, 40, 3), dtype=np.float32)
# for image in os.listdir(path):
#     im = Image.open(path+image)
#     im_arr = np.array(im)
#     im_array = np.vstack((im_array, im_arr))
#
# np.save('neg.npy', im_array)
# img = np.load('neg.npy')
# print img.shape


# new_im = Image.fromarray(imarray)
# new_im.show()
