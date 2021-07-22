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

def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

def target_distribution(q):
    p = torch.nn.functional.one_hot(torch.argmax(q, dim=1), q.shape[1]).to(dtype=torch.float32)
    return p

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

class DGAE(nn.Module):

    def __init__(self, **kwargs):
        super(DGAE, self).__init__()
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
                                      
    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs, lr, save_path, dataset):
        if  not os.path.exists(save_path + dataset + '/pretrain/model.pk'):
            if optimizer == "Adam":
                opti = Adam(self.parameters(), lr=lr)
            elif optimizer == "SGD":
                opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.001)
            elif optimizer == "RMSProp":
                opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.001)
            print('Pretraining......')
            
            # initialisation encoder weights
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
                cm = clustering_metrics(y, y_pred)
                acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
                if (acc > acc_best):
                    acc_best = acc
                    torch.save(self.state_dict(), save_path + dataset + '/pretrain/model.pk')
                print("Best accuracy : ", acc_best)
        else:
            self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
      
    def loss(self, q, p, x_, adj_label, weight_tensor, norm):
        loss_recons = F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight=weight_tensor)
        loss_clus = self.kl_loss(torch.log(q), p)
        loss = loss_recons +  self.gamma * loss_clus
        print("loss clustering: " + str(loss_clus))
        print("loss reconstruction: " + str(loss_recons))
        print("loss: " + str(loss))
        return loss, loss_recons, loss_clus 
    
    def train(self, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs, lr, save_path, dataset):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay = 0.001)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        
        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        logfile = open(save_path + dataset + '/cluster/log_DGAE_' + dataset + '.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'loss_recons', 'loss_clus' , 'loss'])
        logwriter.writeheader()
        epoch_bar = tqdm(range(epochs))
        km = KMeans(n_clusters=self.nClusters, n_init=20)

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
            if epoch % 5 == 0:
                p = target_distribution(q.detach())
            x_ = self.decode(emb)
            loss, loss_recons, loss_clus = self.loss(q, p, x_, adj_label, weight_tensor, norm)  
            epoch_bar.write('Loss={:.4f}'.format(loss.detach().numpy()))
            y_pred = self.predict(emb)                            
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            
            # Update learnable parameters
            loss.backward()
            opti.step()
            lr_s.step()

            #Save logs            
            logdict = dict(iter = epoch, acc = acc, nmi= nmi, ari=adjscore, f1_macro = f1_macro , f1_micro = f1_micro, precision_macro = precision_macro, precision_micro = precision_micro, loss_recons = loss_recons.detach().numpy(), loss_clus = loss_clus.detach().numpy(), loss = loss.detach().numpy())
            logwriter.writerow(logdict)
            logfile.flush()

        return y_pred

    def predict(self, emb):
        with torch.no_grad():
            q = self.assignment(emb)
        return np.argmax(q.detach().numpy(), axis=1)

    def encode(self, x_features, adj):
        hidden = self.gcn_1(x_features, adj)
        self.embedded = self.gcn_2(hidden, adj)
        return self.embedded

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred