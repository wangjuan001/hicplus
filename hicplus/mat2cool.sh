dat=$1 ##output from hicplus prediction
chrom=mm10.chrom.sizes ##chrom size file, change to your own species. 

cat $dat | tr ':' '\t'|tr '-' '\t' > ${dat}_tmp

###transfrom the matrix file to .cool file
cooler load -f bg2 ${chrom}:10000 ${dat}_tmp ${dat}.cool  --input-copy-status duplex

## remove the intermediate tmp file. 
rm ${dat}_tmp
