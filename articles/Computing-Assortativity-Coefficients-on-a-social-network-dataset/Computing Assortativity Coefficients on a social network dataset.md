---
title: Computing Assortativity Coefficients on a social network dataset.
author: Gabriele Albini
date:
topics: [Social Networks Analysis, Graphs, Assortativity]
meta: Upload and visualize Facebook Network data, compute the assortativity coefficient and understand its relevance.
target: Midway
language: English
cover: cover.png
---

# Computing Assortativity Coefficients on a social network dataset

In this article we will use some Facebook data to explore the concept of network assortativity (also called as homophily), which we define as the tendency of nodes to connect to their similar.
Networks or Graphs are data representation consisting in nodes (vertices) and edges (links): in this article we will consider only undirected and unweighted edges. We will first of all present the dataset we intend to use, going through the data loading and wrangling steps and presenting the network.
Next, we will introduce the concept of network assortativity. The main theoretical framework which will be used is the article from Newman et al. “Mixing patterns in networks”, 2003 (available [here](https://arxiv.org/abs/cond-mat/0209450)) which defines and explains the concept of network assortativity.
We will then apply this metric to the dataset, in order to confirm whether — as stated in the article – people tend to connect to others who are like them.

## 1. The data
The data (anonymized and publicly available) used in the article can be downloaded from [this page](https://snap.stanford.edu/data/ego-Facebook.html). We will need two sets of files:

1. the [file](https://snap.stanford.edu/data/facebook_combined.txt.gz) “facebook_combined.txt.gz” contains edges from 4039 nodes of 10 networks. Edges are represented in an adjacency list format (i.e. [0,1] means there’s an edge between node 0 and node 1).
2. the [file](https://snap.stanford.edu/data/facebook.tar.gz) “facebook.tar.gz” contains several other files. We will be using only “.feat” and “.featnames” files, which corresponds to network attributes (and their names) for all the nodes.

### 1.1 Loading the network
We will be using the NetworkX library in Python.

Importing the main network file is very easy:
``` python
_path = r'Downloads/facebook_combined.txt'
G_fb = nx.read_edgelist(_path, create_using = nx.Graph(), nodetype=int)
```
