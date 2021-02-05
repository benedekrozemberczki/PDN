PDN
============================================
[![Arxiv](https://img.shields.io/badge/ArXiv-2010.12878-orange.svg?color=blue)](https://arxiv.org/abs/2010.12878) [![codebeat badge](https://codebeat.co/badges/f7212651-50c6-40bd-9f4c-030ea56f43d3)](https://codebeat.co/projects/github-com-benedekrozemberczki-pdn-master)
 [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/PDN.svg)](https://github.com/benedekrozemberczki/PDN/archive/master.zip)⠀

A PyTorch implementation of "Pathfinder Discovery Networks for Neural Message Passing" (WebConf 2021).

### Abstract

<p align="justify">
Graph convolutional network (GCN) has been successfully applied to many graph-based applications; however, training a large-scale GCN remains challenging. Current SGD-based algorithms suffer from either a high computational cost that exponentially grows with number of GCN layers, or a large space requirement for keeping the entire graph and the embedding of each node in memory. In this paper, we propose Cluster-GCN, a novel GCN algorithm that is suitable for SGD-based training by exploiting the graph clustering structure. Cluster-GCN works as the following: at each step, it samples a block of nodes that associate with a dense subgraph identified by a graph clustering algorithm, and restricts the neighborhood search within this subgraph. This simple but effective strategy leads to significantly improved memory and computational efficiency while being able to achieve comparable test accuracy with previous algorithms. To test the scalability of our algorithm, we create a new Amazon2M data with 2 million nodes and 61 million edges which is more than 5 times larger than the previous largest publicly available dataset (Reddit). For training a 3-layer GCN on this data, Cluster-GCN is faster than the previous state-of-the-art VR-GCN (1523 seconds vs 1961 seconds) and using much less memory (2.2GB vs 11.2GB). Furthermore, for training 4 layer GCN on this data, our algorithm can finish in around 36 minutes while all the existing GCN training algorithms fail to train due to the out-of-memory issue. Furthermore, Cluster-GCN allows us to train much deeper GCN without much time and memory overhead, which leads to improved prediction accuracy -- using a 5-layer Cluster-GCN, we achieve state-of-the-art test F1 score 99.36 on the PPI dataset, while the previous best result was 98.71.</p>

This repository provides a PyTorch implementation of ClusterGCN as described in the paper:

> Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
> Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, Cho-Jui Hsieh.
> KDD, 2019.
> [[Paper]](https://arxiv.org/abs/1905.07953)
