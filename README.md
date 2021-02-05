PDN
============================================


[![Arxiv](https://img.shields.io/badge/ArXiv-2010.12878-orange.svg?color=blue)](https://arxiv.org/abs/2010.12878) [![codebeat badge](https://codebeat.co/badges/f7212651-50c6-40bd-9f4c-030ea56f43d3)](https://codebeat.co/projects/github-com-benedekrozemberczki-pdn-master)
 [![repo size](https://img.shields.io/github/repo-size/benedekrozemberczki/PDN.svg)](https://github.com/benedekrozemberczki/PDN/archive/master.zip)⠀
 
 A PyTorch implementation of "Pathfinder Discovery Networks for Neural Message Passing" (WebConf 2021).


<p align="center">
  <img width="600" src="pdn.jpeg">
</p>

### Abstract

<p align="justify">
In this work we propose Pathfinder Discovery Networks (PDNs), a method for jointly learning a message passing graph over a multiplex network with a downstream semi-supervised model. PDNs inductively learn an aggregated weight for each edge, optimized to produce the best outcome for the downstream learning task. PDNs are a generalization of attention mechanisms on graphs which allow flexible construction of similarity functions between nodes, edge convolutions, and cheap multiscale mixing layers. We show that PDNs overcome weaknesses of existing methods for graph attention (e.g. Graph Attention Networks), such as the diminishing weight problem. Our experimental results demonstrate competitive predictive performance on academic node classification tasks. Additional results from a challenging suite of node classification experiments show how PDNs can learn a wider class of functions than existing baselines. We analyze the relative computational complexity of PDNs, and show that PDN runtime is not considerably higher than static-graph models. Finally, we discuss how PDNs can be used to construct an easily interpretable attention mechanism that allows users to understand information propagation in the graph.</p>

This repository provides a PyTorch implementation of PDN as described in the paper:

> Pathfinder Discovery Networks for Neural Message Passing.
> Benedek Rozemberczki, Peter Englert, Amol Kapoor, Martin Blais, Bryan Perozzi.
> WebConf, 2021.
> [[Paper]](https://arxiv.org/abs/2010.12878)
