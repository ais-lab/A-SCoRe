# A-SCoRe

This repo contains code associated with the A-SCoRe paper

## Installation 
To use pnp ransac, you need to build the cython module:
```shell
cd ./pnpransac/
python setup.py build_ext --inplace
```
Then create the conda environment using the `.yml` config file:
```
conda env create -f environment.yml
conda activate ascore
```

## Dataset

### 7-Scenes 
Download the dataset from the project page
```shell
mkdir datasets/7scenes
export dataset=datasets/7scenes
for scene in chess fire heads office pumpkin redkitchen stairs; \
do wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip -P $dataset \
&& unzip $dataset/$scene.zip -d $dataset && unzip $dataset/$scene/'*.zip' -d $dataset/$scene; done
```

### 12-Scenes

```shell
mkdir datasets/12scenes
export dataset=datasets/12scenes
for scene in apt1 apt2 office1 office2; \
  do wget https://graphics.stanford.edu/projects/reloc/data/$scene.zip -P $dataset \
    && unzip $dataset/$scene.zip -d $dataset;
  done
```

### Cambridge


### Custom
Coming

## Usage
### Training and validation (dense setting)

```shell
python lightning/main.py --model sp_sg_attn \
                          --dataset 7Scenes \ 
                          --scene chess \
                          --device 0 \
                          --mode train \
                          --n_iter 300000 \
                          --batch_size 8 \
                          --num_workers 4 \
                          --aug True \
                          --data_path /path/to/metadata \ 
                          --img_path /path/to/image \
                          
```

## Publication
If you find any part of this code useful, please cite:
```

```

## Acknowledgement

