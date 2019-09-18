## Introduction
This is a supplement for HiCplus, which accept .hic file. Previous HiCplus requires high memory and GPU when predicting to a high resolution matrix(e.g.10kb). Here, we can make high resolution predictions on CPU with relatively low memory requirements (<100G) on whole chromosomes.

### Installation
conda create -n plus python=3.6 numpy pytorch torchvision
source activate plus
git clone -b packageup --single-branch https://github.com/wangjuan001/HiCplus_straw.git
cd HiCplus_straw
python setup.py install

### Usage
input .hic file  (low resolution)

output  .npy file at chromosome level (both inter- and intra-) (enhanced resolution).

Output file will be generated in the current directory named as chr1.chr2.pred.npy.

```
python strawHiCplus.py -h

```

```
usage: strawHiCplus.py [-h] -i INPUT -m MODEL [-b BINSIZE] -c chrN1 chrN2

PyTorch Super Res From .hic file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input .hic file to use
  -m MODEL, --model MODEL
                        model file to use
  -b BINSIZE, --binsize BINSIZE
                        binsize, default:10000
  -c chrN1 chrN2, --chrN chrN1 chrN2
                        chromosome number

```

### Requirements
install HiCplus

install straw
