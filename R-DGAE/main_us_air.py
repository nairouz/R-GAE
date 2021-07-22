#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License
import numpy as np
import torch
import scipy.sparse as sp
from model_brazil_air import ReDGAE
from preprocessing import sparse_to_tuple, preprocess_graph, load_data_networks

# Dataset Name
dataset = "usa"
print("USA dataset")
nClusters = 4
adj, features , labels = load_data_networks('usa', './data/Airports/')
num_nodes = features.shape[1]
num_features = features.shape[0]

# Network parameters
alpha = 1.
gamma = 0.001
num_neurons = 32
embedding_size = 16
save_path = "./results/"

# Pretraining parameters
epochs_pretrain = 200
lr_pretrain = 0.01

# Clustering parameters
epochs_cluster = 200
lr_cluster = 0.00001

beta1 = 0.1
beta2 = 0.06

 # Data processing 
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_norm = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)
adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T), torch.FloatTensor(adj_norm[1]), torch.Size(adj_norm[2]))
adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T), torch.FloatTensor(features[1]), torch.Size(features[2]))
weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig 

# Training
network = ReDGAE(adj=adj_norm, num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation="ReLU", alpha=alpha, gamma=gamma)
#network.pretrain(adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_pretrain, lr=lr_pretrain, save_path=save_path, dataset=dataset)
network.train(adj_norm, features, adj, adj_label, labels, weight_tensor_orig, norm, optimizer="Adam", epochs=epochs_cluster, lr=lr_cluster, beta1=beta1, beta2=beta2, save_path=save_path, dataset=dataset)