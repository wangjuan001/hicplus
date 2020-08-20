#!/usr/bin/env python
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
from hicplus import pred_chromosome

startTime = datetime.now()

def pred_genome(hicfile, binsize, inmodel):
    hic_info = utils.read_hic_header(hicfile)
    chromindex = {}
    i = 0
    for c, Len in hic_info['chromsizes'].items():
        chromindex[c] = i
        i += 1
    print(hic_info)

    name = os.path.basename(inmodel).split('.')[0]
    with open('genome.{}_{}.matrix.txt'.format(int(binsize/1000),name), 'w') as genome:
        for c1, Len1 in hic_info['chromsizes'].items():
            for c2, Len2 in hic_info['chromsizes'].items():
                if chromindex[c1] > chromindex[c2]:
                    continue
                if c1 == 'M' or c2 == 'M':
                    continue
                Mat = pred_chromosome.chr_pred(hicfile, c1, c2, binsize, inmodel)
                r, c = Mat.nonzero()
                for i in range(r.size):
                    contact = int(round(Mat[r[i],c[i]]))
                    if contact == 0:
                        continue
                    if r[i]*binsize > Len1 or (r[i]+1)*binsize > Len1:
                        continue
                    if c[i]*binsize > Len2 or (c[i]+1)*binsize > Len2:
                        continue
                    line = [c1, r[i]*binsize, (r[i]+1)*binsize,
                           c2, c[i]*binsize, (c[i]+1)*binsize, contact]
                    genome.write('chr'+str(line[0])+':'+str(line[1])+'-'+str(line[2])+
                                 '\t'+'chr'+str(line[3])+':'+str(line[4])+'-'+str(line[5])+'\t'+str(line[6])+'\n')




def main(args):
    binsize = args.binsize
    inmodel = args.model
    hicfile = args.inputfile
    pred_genome(hicfile, binsize, inmodel)

if __name__ == '__main__':
    main()

