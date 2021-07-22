#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from torch.nn import Parameter
from sklearn import metrics
from munkres import Munkres
from sklearn.cluster import KMeans
import metrics as mt
import scipy.sparse as sp
from preprocessing import sparse_to_tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def map_vector_to_clusters(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_true[i], y_pred[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    y_true_mapped = np.zeros(y_pred.shape)
    for i in range(y_pred.shape[0]):
        y_true_mapped[i] = col_ind[y_true[i]]
    return y_true_mapped.astype(int)

def q_mat(X, centers, alpha=1.0):
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q**((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return q
  
def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    confidence1 = np.zeros((q.shape[0],))
    confidence2 = np.zeros((q.shape[0],))
    a = np.argsort(q, axis=1)
    for i in range(q.shape[0]):
        confidence1[i] = q[i,a[i,-1]]
        confidence2[i] = q[i,a[i,-2]]
        if (confidence1[i]) > beta1 and (confidence1[i] - confidence2[i]) > beta2:
            unconf_indices.append(i)
        else:
            conf_indices.append(i)
    unconf_indices = np.asarray(unconf_indices, dtype=int)
    conf_indices = np.asarray(conf_indices, dtype=int)
    return unconf_indices, conf_indices

def target_distribution(q):
    p = torch.nn.functional.one_hot(torch.argmax(q, dim=1), q.shape[1]).to(dtype=torch.float32)
    return p

def evaluate_links(adj, labels):
    nb_links = 0
    nb_false_links = 0
    nb_true_links = 0
    k = 0
    for line in adj:
        for j in range(line.indices.size):
            nb_links += 1 
            if labels[k] == labels[line.indices[j]]:
                nb_true_links += 1
            else:
                nb_false_links += 1
        k += 1
    assert nb_links == nb_true_links + nb_false_links
    print(" nb_links: " + str(nb_links) + " nb_true_links: " + str(nb_true_links) + "  nb_false_links: " + str(nb_false_links))  
    return nb_links, nb_false_links , nb_true_links

class clustering_metrics():

    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = random_uniform_init(input_dim, output_dim) 
        self.activation = activation
        
    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class ClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha, cluster_centers=None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, inputs):
        norm_squared = torch.sum((inputs.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class ReDGAE(nn.Module):

    def __init__(self, **kwargs):
        super(ReDGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        self.alpha = kwargs['alpha']
        self.gamma = kwargs['gamma']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh
        #  layers
        self.gcn_1 = GraphConvSparse(self.num_features, self.num_neurons, self.activation)
        self.gcn_2 = GraphConvSparse(self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.assignment = ClusterAssignment(self.nClusters, self.embedding_size, self.alpha)
        self.kl_loss = nn.KLDivLoss(size_average=False)    
                                      
    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path, dataset):
        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay = 0.001)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr)
        
        print('Pretraining......')
            
        epoch_bar = tqdm(range(epochs))
        acc_best = 0
        km = KMeans(n_clusters=self.nClusters, n_init=20)
        for _ in epoch_bar:
            opti.zero_grad()
            z = self.encode(features, adj)
            x_ = self.decode(z)
            loss = F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            loss.backward()
            opti.step()
            epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
            y_pred = km.fit_predict(z.detach().numpy())
            
      
    def loss(self, q, p, x_, adj_label, weight_tensor, norm):
        loss_recons = F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        loss_clus = self.kl_loss(torch.log(q), p)
        loss = loss_recons +  self.gamma * loss_clus
        print("loss clustering: " + str(loss_clus))
        print("loss reconstruction: " + str(loss_recons))
        print("loss: " + str(loss))
        return loss, loss_recons, loss_clus 
    
    def train(self, adj_norm, features, adj, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, beta1, beta2, save_path, dataset):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr , weight_decay = 0.089)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        
        epoch_bar = tqdm(range(epochs))
        km = KMeans(n_clusters=self.nClusters, n_init=20)
        
        print('Training......')
        
        #initialise the cluster centers 
        with torch.no_grad():
            emb = self.encode(features, adj_norm)
            km.fit(emb.detach().numpy())
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float, requires_grad=True) 
            self.assignment.state_dict()["cluster_centers"].copy_(centers)
        epoch_stable = 0 
        previous_unconflicted = []
        previous_conflicted = []
        
        #initialise the cluster centers 
        with torch.no_grad(): 
            emb = self.encode(features, adj_norm)
            km.fit(emb.detach().numpy()) 
            centers = torch.tensor(km.cluster_centers_, dtype=torch.float, requires_grad=True) 
            self.assignment.state_dict()["cluster_centers"].copy_(centers)
        
        for epoch in epoch_bar:
            opti.zero_grad()
            emb = self.encode(features, adj_norm) 
            q = self.assignment(emb)
            if epoch % 15 == 0:
                p = target_distribution(q.detach())
            x_ = self.decode(emb)
            unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(emb.detach().numpy(), self.assignment.cluster_centers.detach().numpy(), beta1, beta2)
            
            if epoch == 0:
                adj, adj_label, weight_tensor = self.update_graph(adj, y, emb, unconflicted_ind)
            
            if len(previous_unconflicted) < len(unconflicted_ind) :
                emb_unconf = emb[unconflicted_ind]
                p_unconf = p[unconflicted_ind]
                q_unconf = q[unconflicted_ind]
                y_unconf = y[unconflicted_ind]
                previous_conflicted = conflicted_ind
                previous_unconflicted = unconflicted_ind
            
            else:
                epoch_stable += 1
                emb_unconf = emb[previous_unconflicted]
                p_unconf = p[previous_unconflicted]
                q_unconf = q[previous_unconflicted]
                y_unconf = y[previous_unconflicted]
                
            if epoch_stable >= 15:
                print("Update beta_1 and beta_2 \n") 
                epoch_stable = 0
                beta1 = beta1 * 0.95 
                beta2 = beta2 * 0.85 
            
            if epoch % 50 == 0 and epoch <=200 :
                adj, adj_label, weight_tensor =  self.update_graph(adj, y, emb, unconflicted_ind)
            
            loss, loss_recons, loss_clus = self.loss(q_unconf, p_unconf, x_, adj_label, weight_tensor, norm)  
            epoch_bar.write('Loss={:.4f}'.format(loss.detach().numpy()))
            y_pred = self.predict(emb)                            
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            
            # Update learnable parameters
            loss.backward()
            opti.step()
            lr_s.step()
           
        return y_pred
               
    def predict(self, emb):
        with torch.no_grad():
            q = self.assignment(emb)
            out = np.argmax(q.detach().numpy(), axis=1)
        return out

    def encode(self, x_features, adj):
        hidden = self.gcn_1(x_features, adj)
        self.embedded = self.gcn_2(hidden, adj)
        return self.embedded

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

    def compute_acc_and_nmi_conflicted_data(self, unconf_indices, conf_indices, emb, y, save_path, dataset):  
        if unconf_indices.size == 0:
            print(' '*8 + "Empty list of unconflicted data")
            acc_unconf = 0
            nmi_unconf = 0
        else:
            print("Number of unconflicted points : ", len(unconf_indices))
            emb_unconf = emb[unconf_indices]
            y_unconf = y[unconf_indices]
            y_pred_unconf = self.predict(emb_unconf)
            acc_unconf = mt.acc(y_unconf, y_pred_unconf)
            nmi_unconf = mt.nmi(y_unconf, y_pred_unconf)
            print(' '*8 + '|==>  acc unconflicted data: %.4f,  nmi unconflicted data: %.4f  <==|'% (acc_unconf, nmi_unconf))

        if conf_indices.size == 0:
            print(' '*8 + "Empty list of conflicted data")
            acc_conf = 0
            nmi_conf = 0
        else:
            print("Number of conflicted points : ", len(conf_indices))
            emb_conf = emb[conf_indices] 
            y_conf = y[conf_indices]
            y_pred_conf = self.predict(emb_conf)
            acc_conf = mt.acc(y_conf, y_pred_conf)
            nmi_conf = mt.nmi(y_conf, y_pred_conf)
            print(' '*8 + '|==>  acc conflicted data: %.4f,  nmi conflicted data: %.4f  <==|'% (mt.acc(y_conf, y_pred_conf), mt.nmi(y_conf, y_pred_conf)))    
        return acc_unconf, nmi_unconf, acc_conf, nmi_conf

    def generate_centers(self, emb_unconf, y_pred):
        nn = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.assignment.cluster_centers.detach().numpy())
        return indices[y_pred]

    def update_graph(self, adj, labels, emb, unconf_indices):
        y_pred = self.predict(emb)
        emb_unconf = emb[unconf_indices]
        adj = adj.tolil()
        idx = unconf_indices[self.generate_centers(emb_unconf, y_pred)]
        for i, k in enumerate(unconf_indices):
            adj_k = adj[k].tocsr().indices
            if not(np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]):
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, unconf_indices) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), torch.FloatTensor(adj_label[1]), torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() 
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor