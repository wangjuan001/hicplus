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
#arr = arr[0:4,].reshape(2*28,2*28)
median = np.median(arr)
arr[arr<=median] = 0
#arr = np.triu(arr)
x, y = np.where(arr!=0)
#arr[y, x] = arr[x, y] ## if it's predicted from intra chromosome
arr = arr * (1/arr[x,y].min())
#vmax = np.percentile(arr[x,y], 95)
#print(arr)
arr[arr<=1] = 1
arr = np.log(arr)
#arr = arr.filled(0)
#MaxValue= np.mean(arr)

#arr = arr-MaxValue/MaxValue
#print(arr)
#arr = np.ma.masked_where(arr < 0.0001, arr)

#arr = arr.astype('uint8')
cmap = plt.get_cmap('Reds')

plt.imshow(arr, interpolation='none', cmap=cmap, vmax=0.5)
plt.colorbar()
plt.savefig('test.png', dpi=300, bbox_inches='tight')

#cmap.set_bad(color = 'white')

#rgba_arr = cmap(arr)
#print(rgba_arr)
#rgb_arr = np.delete(rgba_arr,3,2)
#print(rgb_arr.shape)
#rgb_arr =rgb_arr.astype('uint8')
#plt.imsave(InFile+'.jpg',rgb_arr)

#new_im = Image.fromarray(rgb_arr)#.convert('RGB')

#new_im.save('new_im.jpg')
