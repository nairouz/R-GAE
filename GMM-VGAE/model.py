#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Mohamed Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Paper   : Rethinking Graph Autoencoder Models for Attributed Graph Clustering
# @License : MIT License

import os
import torch
import metrics as mt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from munkres import Munkres

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

class GMM_VGAE(nn.Module):
    def __init__(self, **kwargs):
        super(GMM_VGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        if kwargs['activation'] == "ReLU":
            self.activation = F.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = F.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = F.tanh

        # VGAE training parameters
        self.base_gcn = GraphConvSparse( self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse( self.num_neurons, self.embedding_size, activation = lambda x:x)
        
        # GMM training parameters    
        self.pi = nn.Parameter(torch.ones(self.nClusters)/self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size),requires_grad=True)
                                  
    def pretrain(self, adj, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs, lr, save_path, dataset):
        if  not os.path.exists(save_path + dataset + '/pretrain/model.pk'):
            if optimizer == "Adam":
                opti = Adam(self.parameters(), lr=lr)
            elif optimizer == "SGD":
                opti = SGD(self.parameters(), lr=lr, momentum=0.9)
            elif optimizer == "RMSProp":
                opti = RMSprop(self.parameters(), lr=lr)
            print('Pretraining......')
            
            # initialisation encoder weights
            epoch_bar = tqdm(range(epochs))
            acc_best = 0
            gmm = GaussianMixture(n_components = self.nClusters , covariance_type = 'diag')
            acc_list = []
            for _ in epoch_bar:
                opti.zero_grad()
                _,_, z = self.encode(features, adj)
                x_ = self.decode(z)
                loss = norm*F.binary_cross_entropy(x_.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
                loss.backward()
                opti.step()
                epoch_bar.write('Loss pretraining = {:.4f}'.format(loss))
                y_pred = gmm.fit_predict(z.detach().numpy())
                print("pred_gmm : ", y_pred)
                print("Pred unique labels : ", set(y_pred))
                print("Pred length : ", len(y_pred))
                self.pi.data = torch.from_numpy(gmm.weights_)
                self.mu_c.data = torch.from_numpy(gmm.means_)
                self.log_sigma2_c.data =  torch.log(torch.from_numpy(gmm.covariances_))
                acc = mt.acc(y, y_pred)
                acc_list.append(acc)
                if (acc > acc_best):
                  acc_best = acc
                  self.logstd = self.mean 
                  torch.save(self.state_dict(), save_path + dataset + '/pretrain/model.pk')
            print("Best accuracy : ",acc_best)
            return acc_list
        else:
            self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
      
    def ELBO_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c
        det = 1e-2 
        
        Loss_recons = 1e-2 * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight = weight_tensor)
        Loss_recons = Loss_recons * features.size(0)
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(emb, mu_c,log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1,1))
        y_pred = self.predict(emb)
        
        KL1 = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)),2),1))
        KL2 = torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        Loss_clus = KL1 - KL2
        
        Loss_elbo =  Loss_recons + Loss_clus 
        return Loss_elbo, Loss_recons, Loss_clus 
   
    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer="Adam", epochs, lr, save_path, dataset):
        self.load_state_dict(torch.load(save_path + dataset + '/pretrain/model.pk'))
        if optimizer ==  "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay = 0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay = 0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay = 0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)
        
        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        logfile = open(save_path + dataset + '/cluster/log.csv', 'w')
        
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro', 'precision_macro', 'precision_micro', 'Loss_recons', 'Loss_clus' , 'Loss_elbo'])
        
        logwriter.writeheader()
        
        epoch_bar=tqdm(range(epochs))
        
        print('Training......')
        
        loss_list = []
        grad_loss_list = [] 
        for epoch in epoch_bar:
            opti.zero_grad()
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm) 
            x_ = self.decode(emb)
            Loss_elbo, Loss_recons, Loss_clus = self.ELBO_Loss(features, adj_norm, x_, adj_label.to_dense().view(-1), y, weight_tensor, norm, z_mu , z_sigma2_log, emb)
            epoch_bar.write('Loss={:.4f}'.format(Loss_elbo.detach().numpy()))
            y_pred = self.predict(emb)                            
            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore, f1_macro, precision_macro, f1_micro, precision_micro = cm.evaluationClusterModelFromLabel()
            acc_list.append(acc)
            
            #Save logs 
            logdict = dict(iter = epoch, acc = acc, nmi= nmi, ari=adjscore, f1_macro=f1_macro , f1_micro=f1_micro, precision_macro=precision_macro, precision_micro = precision_micro, Loss_recons=Loss_recons.detach().numpy(), Loss_clus=Loss_clus.detach().numpy(), Loss_elbo=Loss_elbo.detach().numpy())
            logwriter.writerow(logdict)
            logfile.flush() 
            
            Loss_elbo.backward()
            opti.step()
            lr_s.step()
        return acc_list, y_pred, y
               
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
        return self.mean, self.logstd, sampled_z
            
    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z,z.t()))
        return A_pred
        
def random_uniform_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)
  
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