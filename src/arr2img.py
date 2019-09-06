#from __future__ import division
import numpy as np
from PIL import Image
import sys
import os
import gzip
import matplotlib.pyplot as plt
from numpy import *

InFile = sys.argv[1] ## input .npy file

HiC_max_value = 100
arr = np.load(InFile).astype(np.float32)
#print(arr)
arr = ma.log(arr)+2
arr = arr.filled(0)
MaxValue= np.mean(arr)

arr = arr-MaxValue/MaxValue
#print(arr)
#arr = np.ma.masked_where(arr < 0.0001, arr)

#arr = arr.astype('uint8')
cmap = plt.get_cmap('Reds')
#cmap.set_bad(color = 'white')

rgba_arr = cmap(arr)
#print(rgba_arr)
rgb_arr = np.delete(rgba_arr,3,2)
#print(rgb_arr.shape)
#rgb_arr =rgb_arr.astype('uint8')
plt.imsave(InFile+'.jpg',rgb_arr)

#new_im = Image.fromarray(rgb_arr)#.convert('RGB')

#new_im.save('new_im.jpg')
