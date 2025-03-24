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
We also provide the SfM models running SuperPoint configuration in [this location]()

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

```
mkdir datasets/Cambridge
export scenes=( "KingsCollege" "OldHospital" "StMarysChurch" "ShopFacade" "GreatCourt" )
export IDs=( "251342" "251340" "251294" "251336" "251291" )
for i in "${!scenes[@]}"; do
wget https://www.repository.cam.ac.uk/bitstream/handle/1810/${IDs[i]}/${scenes[i]}.zip -P $dataset \
&& unzip $dataset/${scenes[i]}.zip -d $dataset && rm $dataset/${scenes[i]}.zip; done
```


### Custom
Coming

## Usage
### Training and validation (dense setting)
Training command:

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
Evaluating command:

```shell
python lightning/main.py --model sp_sg_attn \ 
                          --dataset 7Scenes \
                          --scene chess \
                          --mode test \
                          --device 0 \
                          --n_iter 300000 \ 
                          --batch_size 8 \ 
                          --num_workers 4 \
                          --aug True \
                          --data_path /path/to/metadata \ 
                          --img_path /path/to/image \ 
                          --pretrained_ckpt /path/to/pretrained/model \
                          --validate_on_train False \

```

### Training and validation (sparse setting)
Training command

```shell
python lightning/main.py --model sp_sg_attn \
                          --dataset 7Scenes_SfM \ 
                          --loss regcoord_loss \
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

Evaluating command

```shell
python lightning/main.py --model sp_sg_attn \
                          --dataset 7Scenes_SfM \ 
                          --loss regcoord_loss \
                          --scene chess \
                          --mode test \
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
Thanks to authors of these great opensource projects: [Hloc](https://github.com/cvg/Hierarchical-Localization), [HSCNet](https://github.com/AaltoVision/hscnet), [D2S](https://github.com/ais-lab/d2s) for their contributions that we take inspiration from.
