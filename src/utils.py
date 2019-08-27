import numpy as np
#import matplotlib.pyplot as plt
import os
import random

def readFiles(filename, total_length, resolution):
    print("reading HiC input data")
    infile = open(filename).readlines()
    if len(infile[1].split("\t")) == 3:
        res = readSparseMatrix(infile, total_length, resolution)
        np.savetxt(filename+'square.txt', res, delimiter = '\t', fmt = '%d')
    else:
        res = readSquareMatrix(infile, total_length)
    return res

def readSparseMatrix(infile, total_length, resolution):
    #print "reading Rao's HiC "
    #infile = open(filename).readlines()
    #print len(infile)
    HiC = np.zeros((total_length, total_length)).astype(np.int16)
    percentage_finish = 0
    for i in range(0, len(infile)):
        if (i % (len(infile) / 10) == 0):
            print('finish ', percentage_finish, '%')
            percentage_finish += 10
        nums = infile[i].split('\t')
        try: 
            x = int(int(nums[0])/resolution)
            y = int(int(nums[1])/resolution)
            val = int(float(nums[2]))
        except ValueError:
            pass
        HiC[x][y] = val
        HiC[y][x] = val
    print(HiC.shape)
    return HiC


def readSquareMatrix(infile, total_length):
    #print "reading Rao's HiC "
    #infile = open(filename).readlines()
    #print('size of matrix is ' + str(len(infile)))
    print('number of the bins based on the length of chromsomes is ' + str(total_length))
    result = []
    for line in infile:
        tokens = line.split('\t')
        line_num = list(map(float, tokens))
        line_int = list(map(round, line_num))
        result.append(line_int)
    result = np.array(result)
    print(result.shape)
    return result


def divide(HiCmatrix,chrN):
    subImage_size = 40
    step = 25
    result = []
    index = []
#    chrN = 21  ##need to change.

    total_loci = HiCmatrix.shape[0]
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCmatrix[i:i + subImage_size, j:j + subImage_size]

            result.append([subImage, ])
            tag = 'test'
            index.append((tag, chrN, i, j))
    result = np.array(result)
    print(result.shape)
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
