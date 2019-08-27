import sys
import straw
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

binsize=1000000
result=straw.straw("NONE", "../data/P298plus-DpnII-allReps-filtered.hic","19:1:5000000","19:5000000:10000000","BP",binsize)
#print(result[1])
row = [r//binsize for r in result[0]]
col = [c//binsize for c in result[1]]
value = result[2]
N = max(col) + 1
#print(col)
M = csr_matrix((value, (row,col)), shape=(N,N))
M = csr_matrix.todense(M)

rowix = range(1//1000000,5000000//1000000+1)
colix = range(5000000//1000000,10000000//1000000+1)
M = M[np.ix_(rowix, colix)]
#M[col,row] = M[row,col] ##convert to symmetric 2d matrix
print(M)
