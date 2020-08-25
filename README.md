## Introduction
This is a package for HiCplus, which requires .hic file to train CNN model and with this model you can enhance the resolution of your hic data. High memory and GPU are not necessary when predicting to a high resolution matrix(e.g.10kb).

## Citation
Yan Zhang, Lin An, Jie Xu, Bo Zhang, W. Jim Zheng, Ming Hu, Jijun Tang & Feng Yue. Enhancing Hi-C data resolution with deep convolutional neural network HiCPlus. https://doi.org/10.1038/s41467-018-03113-2.  

### Installation
```
conda config --add channels pytorch  
conda create -n plus python=3.6 numpy pytorch torchvision scipy
python3 -m pip install hic-straw  
source activate plus  
git clone https://github.com/wangjuan001/hicplus.git  
cd hicplus
python setup.py install  
```

### Usage
```
hicplus

usage: hicplus [-h] {train,pred_chromosome} ...

Train CNN model with Hi-C data and make predictions for low resolution HiC
data with the model.

positional arguments:
  {train,pred_chromosome}
    train               Train CNN model per chromosome
    pred_chromosome     predict high resolution interaction frequencies for
                        inter and intra chromosomes

optional arguments:
  -h, --help            show this help message and exit

```

HiCplus training process requires GPU nodes.
```
hicplus train

usage: hicplus train [-h] [-i INPUTFILE] [-r SCALERATE] [-c CHROMOSOME]
                     [-o OUTMODEL]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputfile INPUTFILE
                        path to a .hic file.
  -r SCALERATE, --scalerate SCALERATE
                        downsampling rate to generate the low resolution
                        training file
  -c CHROMOSOME, --chromosome CHROMOSOME
                        choose one chromosome to do the model training.
  -o OUTMODEL, --outmodel OUTMODEL
                        output model name. default = model_epochnumber.model

```
e.g.
```
hicplus train -i https://hicfiles.s3.amazonaws.com/hiseq/gm12878/in-situ/combined.hic -r 40 -c 19
```
You can do prediction on CPUs now.
```
hicplus pred_chromosome
usage: hicplus pred_chromosome [-h] [-i INPUTFILE] [-m MODEL] [-b BINSIZE] -c
                               chrN1 chrN2

optional arguments:
  -h, --help            show this help message and exit
  -i INPUTFILE, --inputfile INPUTFILE
                        path to a .hic file.
  -o OUTPUTFILE, --outputfile OUTPUTFILE
                        path to an output file.
  -m MODEL, --model MODEL
                        path to a model file.
  -b BINSIZE, --binsize BINSIZE
                        predicted resolustion, e.g.10kb, 25kb...,
                        default=10000
  -c chrN1 chrN2, --chrN chrN1 chrN2
                        chromosome number
```
e.g.
```
hicplus pred_chromosome -i test.hic -o test.out.txt -m ../HiCplus_straw/model/pytorch_HindIII_model_40000 -c 19 22 
```

## Transform output matrix to .cool file
To transform the output matrix to .cool file, you can refer to the bash script hicplus/mat2cool.sh. To run the script, you can take the following steps:
1. Specify the chromosome size file for the species you work on. In the example, we used the mm10.chrom.sizes and the related file is provided as well.
2. As the transform is based on cooler https://github.com/mirnylab/cooler, install cooler first if you haven't done so. 
```
conda install -c conda-forge -c bioconda cooler
```
3. After the above setting, you should be able to run the script:
```
sh mat2cool.sh test.out.txt
```

### Model
It's important to use a suitable model when doing prediction. At this moment we only provide one model, which is suitable for 200~300M reads hic data (downsampling rate at 16).   

For other sequencing depth data, the users need to train models at a different downsampling rate (e.g. 40). For more information about how to select downsampling rate, please refer to the original HiCplus paper.
