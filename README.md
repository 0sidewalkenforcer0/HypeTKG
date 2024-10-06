### HypeTKG

This is the code for our EMNLP 2024 paper: 
Temporal Fact Reasoning over Hyper-Relational Knowledge Graphs (https://arxiv.org/abs/2307.10219).

### Installation
## Create a conda environment

```
conda create -n HypeTKG python=3.8
conda activate HypeTKG
```

## Configure HypeTKG requirements

Install PyTorch (specify your own CUDA version)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```
Install PyTorch Geometric:
`
conda install pyg -c pyg
`

Install PyTorch Scatter:
`
conda install pytorch-scatter -c pyg
`

Install wandb:
`
conda install -c conda-forge wandb
`

install tqdm:
`
conda install -c conda-forge tqdm
`

install numpy:
conda install numpy=1.19


### Datasets

The dataset can be found in `data/Wiki-hy` and `data/YAGO-hy`.

Their derivatives can be found there as well:
* `Wiki-hy(33)` and `YAGO-hy(33)` - approx 33% of statements have qualifiers
* `Wiki-hy(66)` and `YAGO-hy(66)` - approx 66% of statements have qualifiers
* `Wiki-hy(100)` and `YAGO-hy(100)` - 100% of statements have qualifiers

### Running Experiments
## Parameters
Parameters are available in the `CONFIG` dictionary in the `run.py`.

* `DATASET`: `YAGO-hy` or `Wiki-hy` 
* `SAMPLER_W_QUALIFIERS`: `True` for hyper-relational models [default], `False` for quadruple-based model only
* `SAMPLER_W_STATICS`:`True` for the model with time-invariant facts, `False` for Hyper-Relational Temporal Knowledge Graphs model only

## Training Model
```
python run.py
```
