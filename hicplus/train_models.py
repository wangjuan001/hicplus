from __future__ import print_function
import argparse as ap
from math import log10

#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
from hicplus import utils
#import model
import argparse
from hicplus import trainConvNet
import numpy as np

chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]

#chrN = 21
#scale = 16

def main(args):

    highres = utils.matrix_extract(args.chromosome, 10000, args.inputfile)

    print('dividing, filtering and downsampling files...')

    highres_sub, index = utils.divide(highres)

    print(highres_sub.shape)
    #np.save(infile+"highres",highres_sub)

    lowres = utils.genDownsample(highres,1/float(args.scalerate))
    lowres_sub,index = utils.divide(lowres)
    print(lowres_sub.shape)
    #np.save(infile+"lowres",lowres_sub)

    print('start training...')
    trainConvNet.train(lowres_sub,highres_sub,args.outmodel)


    print('finished...')
