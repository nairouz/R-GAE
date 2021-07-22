#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License

import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from preprocessing import sparse_to_tuple
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from munkres import Munkres


def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def q_mat(X, centers, alpha=1.0):
    X = X.detach().numpy()
    centers = centers.detach().numpy()
    if X.size == 0:
        q = np.array([])
    else:
        q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(X, 1) - centers), axis=2) / alpha))
        q = q ** ((alpha + 1.0) / 2.0)
        q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
        return q
  
def generate_unconflicted_data_index(emb, centers_emb, beta1, beta2):
    unconf_indices = []
    conf_indices = []
    q = q_mat(emb, centers_emb, alpha=1.0)
    confidence1 = q.max(1)
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
            # correponding label in l2:
            c2 = l2[indexes[i][1]]
            

            # ai is the index with label==c2 in the pred_label list
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
        x = torch.mm(x,self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

class ReGMM_VGAE(nn.Module):
    def __init__(self, **kwargs):
        super(ReGMM_VGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        
        # VGAE training parameters
        self.base_gcn = GraphConvSparse( self.num_features, self.num_neurons)
        self.gcn_mean = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        
        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
                                  
    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, epochs, lr, save_path, dataset):
        opti = Adam(self.parameters(), lr=lr)
        epoch_bar = tqdm(range(epochs))
        gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
        for _ in epoch_bar:
            opti.zero_grad()
            _,_, z = self.encode(features, adj)
            x_ = self.decode(z)
            loss = norm*F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
            loss.backward()
            opti.step()
        gmm.fit_predict(z.detach().numpy())
        self.pi.data = torch.from_numpy(gmm.weights_)
        self.mu_c.data = torch.from_numpy(gmm.means_)
        self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_))
        self.logstd = self.mean 
                                  
    def ELBO_Loss(self, features, adj, x_, adj_label, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2
        Loss = 1e-2 * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor)
        Loss = Loss * features.size(0) 
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(emb,mu_c,log_sigma2_c))+det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        KL1 = 0.5 * torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        Loss1 = KL1 
        KL2= torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))
        Loss1 -= KL2
        return Loss, Loss1, Loss+Loss1

    def generate_centers(self, emb_unconf):
        y_pred = self.predict(emb_unconf)
        nn = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(emb_unconf.detach().numpy())
        _, indices = nn.kneighbors(self.mu_c.detach().numpy())
        return indices[y_pred] 

    def update_graph(self, adj, labels, emb, unconf_indices, conf_indices):
        k = 0
        y_pred = self.predict(emb)
        emb_unconf = emb[unconf_indices]
        adj = adj.tolil()
        idx = unconf_indices[self.generate_centers(emb_unconf)]    
        for i, k in enumerate(unconf_indices):
            adj_k = adj[k].tocsr().indices
            if not(np.isin(idx[i], adj_k)) and (y_pred[k] == y_pred[idx[i]]) :
                adj[k, idx[i]] = 1
            for j in adj_k:
                if np.isin(j, unconf_indices) and (np.isin(idx[i], adj_k)) and (y_pred[k] != y_pred[j]):
                    adj[k, j] = 0
        adj = adj.tocsr()
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T), 
                                    torch.FloatTensor(adj_label[1]),
                                    torch.Size(adj_label[2]))
        weight_mask = adj_label.to_dense().view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() 
        weight_tensor[weight_mask] = pos_weight_orig
        return adj, adj_label, weight_tensor

    def train(self, adj_norm, adj, features, y, norm, epochs, lr, beta1, beta2, save_path, dataset):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        opti = Adam(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        import os, csv
        epoch_bar = tqdm(range(epochs))
        previous_unconflicted = []
        previous_conflicted = []
        epoch_stable = 0   
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            x_ = self.decode(emb)
            if epoch % 1 == 0 :
                unconflicted_ind, conflicted_ind = generate_unconflicted_data_index(emb, self.mu_c, beta1, beta2)
                if epoch == 0:
                    adj, adj_label, weight_tensor = self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            if len(previous_unconflicted) < len(unconflicted_ind) :
                z_mu = z_mu[unconflicted_ind]
                z_sigma2_log = z_sigma2_log[unconflicted_ind]
                emb_unconf = emb[unconflicted_ind]
                emb_conf = emb[conflicted_ind]
                previous_conflicted = conflicted_ind
                previous_unconflicted = unconflicted_ind 
            else :
                epoch_stable += 1
                z_mu = z_mu[previous_unconflicted]
                z_sigma2_log = z_sigma2_log[previous_unconflicted]
                emb_unconf = emb[previous_unconflicted]
                emb_conf = emb[previous_conflicted]
            
            if epoch_stable >= 15:
                epoch_stable = 0
                beta1 = beta1 * 0.94
                beta2 = beta2 * 0.83  
            
            if epoch % 50 == 0 and epoch <= 200 :
                adj, adj_label, weight_tensor = self.update_graph(adj, y, emb, unconflicted_ind, conflicted_ind)
            loss, loss1, elbo_loss = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), weight_tensor, norm, z_mu , z_sigma2_log, emb_unconf)
            epoch_bar.write('Loss={:.4f}'.format(elbo_loss.detach().numpy()))
            y_pred = self.predict(emb)                            
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            
            elbo_loss.backward()
            opti.step()
            lr_s.step()
    
    def gaussian_pdfs_log(self,x,mus,log_sigma2s):
        G=[]
        for c in range(self.nClusters):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
        return torch.cat(G,1)

    def gaussian_pdf_log(self,x,mu,log_sigma2):
        c = -0.5 * torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1)
        return c

    def predict(self, z):
        pi = self.pi
        log_sigma2_c = self.log_sigma2_c  
        mu_c = self.mu_c
        det = 1e-2
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det
        yita = yita_c.detach().numpy()
        return np.argmax(yita, axis=1)

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd ,sampled_z
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred