import os,sys
from torch.utils import data
from hicplus import model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse
import numpy as np
from hicplus import utils
from time import gmtime, strftime
from datetime import datetime
import argparse

startTime = datetime.now()


chrs_length = [0,249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

use_gpu = 0 #opt.cuda
#if use_gpu and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

def prediction(M,N,inmodel):
    low_resolution_samples, index = utils.divide(M)

    batch_size = low_resolution_samples.shape[0] #256

    lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
    lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)


    hires_loader = lowres_loader

    m = model.Net(40, 28)
    if torch.cuda.is_available():
        m.load_state_dict(torch.load(inmodel)).cuda()
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


    #print('predicted sample: ', y_predict.shape, ')  #; index shape is: ', index.shape)
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

def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):
    Step = 20000000
    laststart1 =  chrs_length[chrN1]//Step*Step + Step
    lastend1 = chrs_length[chrN1]
    laststart2 =  chrs_length[chrN2]//Step*Step + Step
    #print(laststart1, chrs_length[chrN2])
    lastend2 = chrs_length[chrN2]
    laststart = max(laststart1, laststart2)
    shiftsize=12*binsize
    chrh = np.array([])
    for start1 in range(1, laststart, Step):
        chrv = np.array([])
        for start2 in range(1, laststart, Step):
            #if chrN1 == chrN2 and start2 < start1:
            #    continue
            M,N = utils.frag_matrix_extract(hicfile, chrN1, chrN2, binsize, start1, start2, lastend1, lastend2, shiftsize, Step)
            #print(N)

            #low_resolution_samples, index = divide(M)
            #print(low_resolution_samples.shape)
            enhM = prediction(M,  N, inmodel)
            #print(enhM.shape)
            senhM = sparse.csr_matrix(enhM)
            chrv = hstack([chrv, senhM]) if chrv.size else senhM#.toarray()
            #print(chrv.shape)
        chrh = vstack([chrh, chrv]) #if chrh.size else chrv#.toarray()
    #chrh = chrh.toarray()
    return(chrh)
    #chrh.toarray()

def main(args):
    chrN1, chrN2 = args.chrN
    binsize = args.binsize
    inmodel = args.model
    hicfile = args.inputfile
    Mat = chr_pred(hicfile,chrN1,chrN2,binsize,inmodel).toarray()
    print(Mat.shape)
    np.save('chr%s.chr%s.pred.npy'%(chrN1,chrN2), Mat)
        #print(enhM.shape)
if __name__ == '__main__':
    main()

print(datetime.now() - startTime)
