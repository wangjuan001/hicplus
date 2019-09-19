import numpy as np
#import matplotlib.pyplot as plt
import os,struct
import random
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np

chrs_length = [0,249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

def matrix_extract(chrN1, binsize, hicfile):

    result = straw.straw('NONE', hicfile, str(chrN1),str(chrN1),'BP',binsize)
    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    value = result[2]
    N = max(max(row)+1, max(col) + 1)
    #print(N)
    M = csr_matrix((value, (row,col)), shape=(N,N))
    M = csr_matrix.todense(M)
    M = np.array(M)
    x, y = np.where(M!=0)
    M[y, x] = M[x, y]
    #rowix = range(start1//binsize, end1//binsize+1)
    #colix = range(start2//binsize, end2//binsize+1)
    #print(rowix,colix)
    #M = M[np.ix_(rowix, colix)]
    #N = M.shape[1]
    return(M)

def frag_matrix_extract(hicfile, chrN1, chrN2, binsize, start1, start2, lastend1, lastend2, shiftsize,Step):

    end1=start1+Step + shiftsize
    end2=start2+Step + shiftsize
    #if end1 > lastend1:
    #    end1 = lastend1
    #if end2 > lastend2:
    #    end2 = lastend2
    result = straw.straw('NONE', hicfile, str(chrN1),str(chrN2),'BP',binsize)
    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    value = result[2]

    N = max(chrs_length[chrN2]//binsize+Step//binsize, chrs_length[chrN1]//binsize+Step// binsize) +1
    #N = max(max(row)+1, max(col) + 1)
    #print(N)
    M = csr_matrix((value, (row,col)), shape=(N,N))
    M = csr_matrix.todense(M)
    rowix = range(start1//binsize, end1//binsize+1)
    colix = range(start2//binsize, end2//binsize+1)
    #print(rowix,colix)
    M = M[np.ix_(rowix, colix)]
    N = M.shape[1]
    return(M,N)

def divide(HiCmatrix):
    subImage_size = 40
    step = 25
    result = []
    index = []
    #chrN = 21  ##need to change.

    total_loci = HiCmatrix.shape[0]
    #print(HiCmatrix.shape)
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCmatrix[i:i + subImage_size, j:j + subImage_size]

            result.append([subImage, ])
            tag = 'test'
            index.append((tag, i, j))
    result = np.array(result)
    #print(result.shape)
    result = result.astype(np.double)
    index = np.array(index)
    return result, index

def genDownsample(original_sample, rate):
    result = np.zeros(original_sample.shape).astype(float)
    for i in range(0, original_sample.shape[0]):
        for j in range(0, original_sample.shape[1]):
            for k in range(0, int(original_sample[i][j])):
                if (random.random() < rate):
                    result[i][j] += 1
    return result


if __name__ == "__main__":
    main()
