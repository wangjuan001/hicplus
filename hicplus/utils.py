import numpy as np
#import matplotlib.pyplot as plt
import os,struct
import random
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np

def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')
        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b

def read_hic_header(hicfile):

    if not os.path.exists(hicfile):
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    magic_string = struct.unpack('<3s', req.read(3))[0]
    req.read(1)
    if (magic_string != b"HIC"):
        return None  # this is not a valid .hic file

    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)

    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)

    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length

    info['chromsizes'] = chromsizes

    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)

    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info

def matrix_extract(chrN1, chrN2, binsize, hicfile):

    result = straw.straw('NONE', hicfile, str(chrN1),str(chrN2),'BP',binsize)

    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    value = result[2]
    Nrow = max(row) + 1
    Ncol = max(col) + 1
    N = max(Nrow, Ncol)

    #print(N)
    M = csr_matrix((value, (row,col)), shape=(N,N))
    M = csr_matrix.todense(M)

    return(M)

    # M = np.array(M)
    # x, y = np.where(M!=0)
    # M[y, x] = M[x, y]
    #rowix = range(start1//binsize, end1//binsize+1)
    #colix = range(start2//binsize, end2//binsize+1)
    #print(rowix,colix)
    #M = M[np.ix_(rowix, colix)]
    #N = M.shape[1]

def divide(HiCmatrix):
    subImage_size = 40
    step = 25

    #chrN = 21  ##need to change.

    total_loci = HiCmatrix.shape[0]
    #print(HiCmatrix.shape)
    for i in range(0, total_loci, step):
        result = []
        index = []
        for j in range(0, total_loci, ):
            if (i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCmatrix[i:i + subImage_size, j:j + subImage_size]

            result.append([subImage, ])
            tag = 'test'
            index.append((tag, i, j))
        result = np.array(result)
    #print(result.shape)
    #result = result.astype(np.double)
        index = np.array(index)
        yield result, index


def train_divide(HiCmatrix):
    subImage_size = 40
    step = 25
    result = []
    index = []
    #chrN = 21  ##need to change.

    total_loci = HiCmatrix.shape[0]
    #print(HiCmatrix.shape)
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (abs(i-j)>201 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
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
