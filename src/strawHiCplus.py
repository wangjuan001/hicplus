import os,sys
from torch.utils import data
import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np
#import utils
from time import gmtime, strftime
from datetime import datetime
import argparse

startTime = datetime.now()

parser = argparse.ArgumentParser(description='PyTorch Super Res From .hic file')
parser.add_argument('-i', '--input', type=str, required=True, help='input .hic file to use')
parser.add_argument('-m','--model', type=str, required=True, help='model file to use')
parser.add_argument('-b','--binsize', type=str, help='binsize, default:10000', default = 10000)
#parser.add_argument('--scale_factor', type=float, help='factor by which super resolution needed')
parser.add_argument('-c','--chrN', nargs=2, metavar=('chrN1','chrN2'), type=int,required=True, help='chromosome number')
#parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
#chrN = '19'
inFile = opt.input
binsize = opt.binsize #10000
inmodel= opt.model#"../model/pytorch_HindIII_model_40000"
Step = 20000000
chrN1, chrN2 = opt.chrN

chrs_length = [0,249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

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
    print(result.shape)
    result = result.astype(np.double)
    index = np.array(index)
    return result, index


def matrix_extract(chrN1,chrN2, binsize, start1, start2, lastend1, lastend2, shiftsize):
    #Step = 20000000
    end1=start1+Step + shiftsize
    end2=start2+Step + shiftsize
    #if end1 > lastend1:
    #    end1 = lastend1
    #if end2 > lastend2:
    #    end2 = lastend2
    result = straw.straw('NONE', inFile, str(chrN1),str(chrN2),'BP',binsize)
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
    print(rowix,colix)
    M = M[np.ix_(rowix, colix)]
    N = M.shape[1]
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
    prediction_2 = prediction_1[6:N-6, 6:N-6]
    return(prediction_2)
    #np.save( 'test.enhanced.npy', prediction_1)


# Allh = np.array([])
# for chrN1 in range(1, 22):
#     Allv = []
#     for chrN2 in range(1,22):
#         if chrN2 < chrN1:
#             continue
#

def chrMatrix_pred(chrN1, chrN2):
    laststart1 =  chrs_length[chrN1]//Step*Step + Step
    lastend1 = chrs_length[chrN1]
    laststart2 =  chrs_length[chrN2]//Step*Step + Step
    #print(laststart1, chrs_length[chrN2])
    lastend2 = chrs_length[chrN2]
    laststart = max(laststart1, laststart2)
    shiftsize=15*binsize
    chrh = np.array([])
    for start1 in range(1, laststart, Step):
        chrv = np.array([])
        for start2 in range(1, laststart, Step):
            #if chrN1 == chrN2 and start2 < start1:
            #    continue
            M,N = matrix_extract(chrN1, chrN2, binsize, start1, start2, lastend1, lastend2, shiftsize )
            #print(N)

            #low_resolution_samples, index = divide(M)
            #print(low_resolution_samples.shape)
            enhM = prediction(M,  N)
            #print(enhM.shape)
            senhM = sparse.csr_matrix(enhM)
            chrv = hstack([chrv, senhM]) if chrv.size else senhM#.toarray()
            #print(chrv.shape)
        chrh = vstack([chrh, chrv]) #if chrh.size else chrv#.toarray()
    chrh = chrh.toarray()
    return(chrh)
    #chrh.toarray()


Mat = chrMatrix_pred(chrN1,chrN2)
print(Mat.shape)
np.save('chrN1%s.chrN2%s.pred.npy'%(chrN1,chrN2), Mat)
        #print(enhM.shape)

print(datetime.now() - startTime)
