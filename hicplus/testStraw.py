import sys
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
import numpy as np
from PIL import Image
import gzip
import matplotlib.pyplot as plt
from numpy import *

np.set_printoptions(threshold=sys.maxsize)

binsize=100000
result=straw.straw("NONE", "/Users/jwn2291/Desktop/strawHiC/HiCplus_straw/data/test.hic","1","12","BP",binsize)
#print(result[1])
row = [r//binsize for r in result[0]]
col = [c//binsize for c in result[1]]
value = result[2]

N = max(max(row)+1, max(col) + 1)
M = csr_matrix((value, (row, col)), shape=(N, N))
#print(M)
M = csr_matrix.todense(M)
print(M.shape)

arr = ma.log(M)+2
arr = arr.filled(0)
cmap = plt.get_cmap('Reds')
rgba_arr = cmap(arr)
rgb_arr = np.delete(rgba_arr,3,2)

plt.imsave('test.jpg',rgb_arr)

rowix = range(1//binsize, 10000000//binsize+1)
colix = range(120000001//binsize, 130000001//binsize+1)
print(rowix,colix)
M = M[np.ix_(rowix, colix)]
print(M.shape)
print(M)
#M = csr_matrix((value, (row,col)), shape=(N,N))
#M = csr_matrix.todense(M)

#rowix = range(1//1000000,5000000//1000000+1)
#colix = range(5000000//1000000,10000000//1000000+1)
#M = M[np.ix_(rowix, colix)]
#M[col,row] = M[row,col] ##convert to symmetric 2d matrix
#print(M)

A = np.random.rand(4,4)
print(A)
A = csr_matrix(A)
print(A)
B = np.random.rand(4,2)
print(B)
A=np.array([])
B = csr_matrix(B)#.toarray()
print(B)
All = hstack([A, B]) #if A.size else B#.toarray()
print(All)

Step = 20000000
chrs_length = [0,249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

laststart1 =  chrs_length[22]//Step*Step + Step
print(laststart1)
shiftsize=15*10000
for start1 in range(1, laststart1, Step):
    #print(start1)
    end1=start1 + Step + shiftsize
    print(end1)


chr1 = "1"
chr2 = "2"
print('chr%s.chr%s.pred.npy'%(chr1,chr2))
