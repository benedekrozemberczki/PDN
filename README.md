PDN
============================================


[![Arxiv](https://img.shields.io/badge/ArXiv-2010.12878-orange.svg?color=blue)](https://arxiv.org/abs/2010.12878) [![codebeat badge](https://codebeat.co/badges/f7212651-50c6-40bd-9f4c-030ea56f43d3)](https://codebeat.co/projects/github-com-benedekrozemberczki-pdn-master)
 [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/PDN.svg)](https://github.com/benedekrozemberczki/PDN/archive/master.zip)⠀[![benedekrozemberczki](https://img.shields.io/twitter/follow/benrozemberczki?style=social&logo=twitter)](https://twitter.com/intent/follow?screen_name=benrozemberczki)
 
 A **PyTorch** implementation of **"Pathfinder Discovery Networks for Neural Message Passing"** (WebConf 2021).
--------------------------------------------

<p align="center">
  <img width="400" src="pdn.jpeg">
</p>

### Abstract

<p align="justify">
In this work we propose Pathfinder Discovery Networks (PDNs), a method for jointly learning a message passing graph over a multiplex network with a downstream semi-supervised model. PDNs inductively learn an aggregated weight for each edge, optimized to produce the best outcome for the downstream learning task. PDNs are a generalization of attention mechanisms on graphs which allow flexible construction of similarity functions between nodes, edge convolutions, and cheap multiscale mixing layers. We show that PDNs overcome weaknesses of existing methods for graph attention (e.g. Graph Attention Networks), such as the diminishing weight problem. Our experimental results demonstrate competitive predictive performance on academic node classification tasks. Additional results from a challenging suite of node classification experiments show how PDNs can learn a wider class of functions than existing baselines. We analyze the relative computational complexity of PDNs, and show that PDN runtime is not considerably higher than static-graph models. Finally, we discuss how PDNs can be used to construct an easily interpretable attention mechanism that allows users to understand information propagation in the graph.</p>

This repository provides a PyTorch implementation of PDN as described in the paper:

> Pathfinder Discovery Networks for Neural Message Passing.
> Benedek Rozemberczki, Peter Englert, Amol Kapoor, Martin Blais, Bryan Perozzi.
> WebConf, 2021.
> [[Paper]](https://arxiv.org/abs/2010.12878)


### Citing

If you find PDN useful in your research, please consider citing the following paper:
```bibtex
>@inproceedings{rozemberczki2021pdn,    
                title={{Pathfinder Discovery Networks for Neural Message Passing}},    
                author={Benedek Rozemberczki and Peter Englert and Amol Kapoor and Martin Blais and Bryan Perozzi},    
                booktitle = {Proceedings of The Web Conference 2021},
                year={2021},    
                organization={ACM}    
                }

```

### Requirements
The codebase is implemented in Python 3.8.5. package versions used for development are just below.
```
tqdm               >=4.50.2
numpy              >=1.19.2
texttable          >=1.6.3
argparse           >=1.1.0
torch              >=1.7.1
torch-geometric    >=1.6.3
torch_spline_conv  >=1.2.0
torch_sparse       >=0.6.8
torch_scatter      >=2.0.5
torch_cluster      >=1.5.8
```

### Options
<p align="justify">
The training of a PDN model is handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options
```
  --edge-path            STR    Edge list NumPy array.        Default is `input/edges.npy`.
  --node-features-path   STR    Node features NumPy array.    Default is `input/node_features.npy`.
  --edge-features-path   STR    Edge features NumPy array.    Default is `input/edge_features.npy`.
  --target-path          STR    Target classes NumPy array.   Default is `input/target.npy`.
```
#### Model options
```
  --seed                INT     Random seed.                   Default is 42.
  --epochs              INT     Number of training epochs.     Default is 200.
  --test-size           FLOAT   Training set ratio.            Default is 0.9.
  --learning-rate       FLOAT   Adam learning rate.            Default is 0.01.
  --edge-filters        INT     Number of PDN filters.         Default is 32.
  --node-filters        INT     Number of GCN filters.         Default is 32.
```
### Examples
<p align="justify">
The following commands learn a neural network and score on the test set. Training a model on the default dataset.</p>

```sh
$ python src/main.py
```
Training a PDN model for a 100 epochs.
```sh
$ python src/main.py --epochs 100
```
Training a model with a different layer structure:
```sh
$ python src/main.py --node-filters 16
```
--------------------------------------------------------------------------------

**License**

- [GNU](https://github.com/benedekrozemberczki/ClusterGCN/blob/master/LICENSE)

--------------------------------------------------------------------------------
