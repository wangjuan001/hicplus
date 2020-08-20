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

use_gpu = 0 #opt.cuda
#if use_gpu and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

def predict(M,N,inmodel):

    prediction_1 = np.zeros((N, N))

    for low_resolution_samples, index in utils.divide(M):

        #print(index.shape)

        batch_size = low_resolution_samples.shape[0] #256

        lowres_set = data.TensorDataset(torch.from_numpy(low_resolution_samples), torch.from_numpy(np.zeros(low_resolution_samples.shape[0])))
        try:
            lowres_loader = torch.utils.data.DataLoader(lowres_set, batch_size=batch_size, shuffle=False)
        except:
            continue

        hires_loader = lowres_loader

        m = model.Net(40, 28)
        m.load_state_dict(torch.load(inmodel, map_location=torch.device('cpu')))

        if torch.cuda.is_available():
            m = m.cuda()

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


        for i in range(0, y_predict.shape[0]):

            x = int(index[i][1])
            y = int(index[i][2])
            #print np.count_nonzero(y_predict[i])
            prediction_1[x+6:x+34, y+6:y+34] = y_predict[i]

    return(prediction_1)

def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel):
    M = utils.matrix_extract(chrN1, chrN2, binsize, hicfile)
    #print(M.shape)
    N = M.shape[0]

    chr_Mat = predict(M, N, inmodel)


#     if Ncol > Nrow:
#         chr_Mat = chr_Mat[:Ncol, :Nrow]
#         chr_Mat = chr_Mat.T
#     if Nrow > Ncol: 
#         chr_Mat = chr_Mat[:Nrow, :Ncol]
#     print(dat.head())       
    return(chr_Mat)



def writeBed(Mat, outname,binsize, chrN1,chrN2):
    with open(outname,'w') as chrom:
        r, c = Mat.nonzero()
        for i in range(r.size):
            contact = int(round(Mat[r[i],c[i]]))
            if contact == 0:
                continue
            #if r[i]*binsize > Len1 or (r[i]+1)*binsize > Len1:
            #    continue
            #if c[i]*binsize > Len2 or (c[i]+1)*binsize > Len2:
            #    continue
            line = [chrN1, r[i]*binsize, (r[i]+1)*binsize,
               chrN2, c[i]*binsize, (c[i]+1)*binsize, contact]
            chrom.write('chr'+str(line[0])+':'+str(line[1])+'-'+str(line[2])+
                     '\t'+'chr'+str(line[3])+':'+str(line[4])+'-'+str(line[5])+'\t'+str(line[6])+'\n')

def main(args):
    chrN1, chrN2 = args.chrN
    binsize = args.binsize
    inmodel = args.model
    hicfile = args.inputfile
    #name = os.path.basename(inmodel).split('.')[0]
    #outname = 'chr'+str(chrN1)+'_'+name+'_'+str(binsize//1000)+'pred.txt'
    outname = args.outputfile
    Mat = chr_pred(hicfile,chrN1,chrN2,binsize,inmodel)
    print(Mat.shape)
    writeBed(Mat, outname, binsize,chrN1, chrN2)
        #print(enhM.shape)
if __name__ == '__main__':
    main()

print(datetime.now() - startTime)
