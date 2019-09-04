import sys
import straw
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

binsize=100000
result=straw.straw("NONE", "../data/test.hic","1","1","BP",binsize)
#print(result[1])
row = [r//binsize for r in result[0]]
col = [c//binsize for c in result[1]]
value = result[2]
N = max(col) + 1
M = csr_matrix((value, (row, col)), shape=(N, N))
M = csr_matrix.todense(M)
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
