import os,sys
from torch.utils import data
import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import straw
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
#import utils
from time import gmtime, strftime
from datetime import datetime

startTime = datetime.now()

#chrN = '19'
binsize= 100000
inmodel="../model/pytorch_HindIII_model_40000"
Step=10000000

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

use_gpu = 0 #opt.cuda
#if use_gpu and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

def divide(HiCmatrix):
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
            index.append((tag, i, j))
    result = np.array(result)
    print(result.shape)
    result = result.astype(np.double)
    index = np.array(index)
    return result, index


def matrix_extract(chrN1,chrN2, binsize, start1, start2):
    Step = 10000000
    end1=start1+Step
    end2=start2+Step
    result = straw.straw('NONE', '../data/test.hic',str(chrN1),str(chrN2),'BP',binsize)
    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    value = result[2]

    N = max(col) + 1
    print(N)
    M = csr_matrix((value, (row,col)), shape=(N,N))
    M = csr_matrix.todense(M)
    rowix = range(start1//binsize, end1//binsize+1)
    colix = range(start2//binsize, end2//binsize+1)
    print(rowix,colix)
    M = M[np.ix_(rowix, colix)]
    print(M.shape)
    return(M,N)


def prediction(M,N):
    low_resolution_samples, index = divide(M)

    batch_size = low_resolution_samples.shape[0] #256

    lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
    lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)


    hires_loader = lowres_loader

    m = model.Net(40, 28)
    m.load_state_dict(torch.load(inmodel, map_location=torch.device('cpu')))

    for i, v1 in enumerate(lowres_loader):
        _lowRes, _ = v1
        _lowRes = Variable(_lowRes).float()
        if use_gpu:
            _lowRes = _lowRes.cuda()
        y_prediction = m(_lowRes)



    y_predict = y_prediction.data.cpu().numpy()

    # recombine samples

    length = int(y_predict.shape[2])
    y_predict = np.reshape(y_predict, (y_predict.shape[0], length, length))


    #length = int(chrs_length[chrN-1]/expRes)

    prediction_1 = np.zeros((N, N))


    print('predicted sample: ', y_predict.shape, ')  #; index shape is: ', index.shape)
    #print index
    for i in range(0, y_predict.shape[0]):
        #if (int(index[i][1]) != chrN):
        #    continue
        #print index[i]
        x = int(index[i][1])
        y = int(index[i][2])
        #print np.count_nonzero(y_predict[i])
        prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]

    return(prediction_1)
    #np.save( 'test.enhanced.npy', prediction_1)


for chrN1 in range(1, 22):
    for chrN2 in range(1,22):
        if chrN2 < chrN1:
            continue
        laststart1 =  chrs_length[chrN1]//Step*Step
        lastend1 = chrs_length[chrN1]
        laststart2 =  chrs_length[chrN2]//Step*Step
        lastend2 = chrs_length[chrN2]
        for start1 in range(1, laststart1, Step):
            for start2 in range(1, laststart2, Step):
                if chrN1 == ChrN2 and start2 < start1:
                    continue
                M,N = matrix_extract(chrN1, chrN2, binsize, start1, start2)
                #print(N)
                end1= start1+Step
                end2= start2+Step
                #print(start1, end1, start2, end2, laststart1, lastend1, laststart2, lastend2)
                if end2 > lastend2 or end1 > lastend1:
                    continue
                print(str(chrN1)+":"+str(start1)+":"+str(end1),str(chrN2)+":"+str(start2)+":"+str(end2))
                low_resolution_samples, index = divide(M)
                print(low_resolution_samples.shape)
                enhM = prediction(M,  N)

        #print(enhM.shape)

print(datetime.now() - startTime)
