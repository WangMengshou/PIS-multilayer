import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing
import time 
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class NeighborFeatureWithEdgeMultiplication(MessagePassing):
    def __init__(self):
        super(NeighborFeatureWithEdgeMultiplication, self).__init__(aggr='mul')  # 使用 'mul' 聚合方式

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, num_features]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, num_features]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: 邻居节点的特征 [num_edges, num_features]
        # edge_attr: 边特征 [num_edges, num_features]
        return 1.0 - x_j * edge_attr  # 将邻居特征与边特征逐元素相乘

    def aggregate(self, inputs, index, dim_size=None):
        # inputs: 消息传递的结果 [num_edges, num_features]
        # index: 目标节点的索引 [num_edges]
        # dim_size: 目标节点的数量
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='mul')  # 使用 'mul' 聚合

class MMCA_threshold():
    def __init__(self):
      self.MessagePassing_model = NeighborFeatureWithEdgeMultiplication()

    def update_r(self, pA, lamM, edge_attr_A, I_edge_index):
        r = ((1 - lamM).unsqueeze(0) * self.MessagePassing_model(pA, I_edge_index, edge_attr_A)).T
        return r
    
    def update_pA(self, pA, r, delta):
        update_pA = (1 - r) + (r-delta) * pA.T
        return update_pA.T
    
    def iteration(self, max_iter, tol, init_pA, epi_paras, soc_paras, I_edge_index):
        lam, delta = epi_paras[...,3], epi_paras[...,4]
        ga0, kgm = soc_paras[:,2], soc_paras[:,5]
        lamM = kgm * ga0 
        edge_attr_A = lam[:,I_edge_index[1]].T
        pA = init_pA.T.clone()
        iter = 0
        err = tol + 1
        while (iter < max_iter) and (err > tol) :
            update_r = self.update_r(pA, lamM, edge_attr_A, I_edge_index)
            update_pA = self.update_pA(pA, update_r, delta)
            err = torch.mean((update_pA - pA) ** 2)
            pA = update_pA
            iter += 1
        return pA

    def mean_pA(self, epi_paras, soc_paras, I_edge_index):
        lam, delta = epi_paras[...,3], epi_paras[...,4]
        ga0, kgm = soc_paras[:,2], soc_paras[:,5]
        lamM = (kgm * ga0).unsqueeze(1)
        degree = torch.mean(torch.bincount(I_edge_index[0]).float())
        b = (lamM + delta - 2 * lam * degree) / (2 * lam * degree)
        c =  - lamM / (2 * lam * degree)
        discriminant = b**2 - 4 * c
        pA = (-b + torch.sqrt(discriminant)) / 2
        return pA.T
    

    def max_eig_H(self, pA, epi_paras, P_matrx):
        rA = epi_paras[...,5]
        max_eig_H = torch.zeros(rA.shape[0])
        for i in range(rA.shape[0]):
            weig = (1-(1-rA) * pA.T).unsqueeze(-1).cpu()
            H = weig[i] * P_matrx
            sparse_matrix = csr_matrix(H)
            eigenvalues, _ = eigs(sparse_matrix, k=1, which='LM')
            max_eig_H[i] = eigenvalues[0]
        return max_eig_H
    
    def max_eig_H_appox(self, pA, epi_paras, P_matrx):
        rA = epi_paras[:,0,5]
        meanpA = torch.mean(pA,dim=0)
        weig = (1-(1-rA) * meanpA).cpu()
        eigenvalues, _ = eigs(P_matrx, k=1, which='LM')
        max_eig_P  = eigenvalues[0]
        max_eig_H_approx = weig * max_eig_P
        return max_eig_H_approx

    def threshold(self, max_eig_H, epi_paras, soc_paras):
        epi_paras = epi_paras.cpu()
        soc_paras = soc_paras.cpu()
        sigma, mu = epi_paras[:,0,1], epi_paras[:,0,2]
        ga0, kgh = soc_paras[:,2], soc_paras[:,6]
        muH = mu + kgh * ga0 - mu * kgh * ga0
        beta_threshold = sigma * muH / (max_eig_H * (sigma + muH) * (1-ga0) )
        return beta_threshold
    

    def degree_MCMC_threshold(self, pA, epi_paras, soc_paras, P_matrix):
        epi_paras = epi_paras.cpu()
        soc_paras = soc_paras.cpu()
        sigma, mu, rA = epi_paras[:,0,1], epi_paras[:,0,2], epi_paras[:,0,5]
        ga0, kgh = soc_paras[:,2], soc_paras[:,6]
        muH = mu + kgh * ga0 - mu * kgh * ga0
        item1 = sigma * muH / ( sigma + muH ) / (1-ga0)
        P_degree = np.sum(P_matrix, axis=1) 
        item2 = np.mean(P_degree)/np.mean(P_degree**2)
        meanpA = torch.mean(pA,dim=0).cpu()
        weig = (1-(1-rA) * meanpA)
        item3 = 1/weig
        beta_threshold = item1 * item2 *item3
        return beta_threshold
    
    def mean_MCMC_threshold(self, pA, epi_paras, soc_paras, P_matrix):
        epi_paras = epi_paras.cpu()
        soc_paras = soc_paras.cpu()
        sigma, mu, rA = epi_paras[:,0,1], epi_paras[:,0,2], epi_paras[:,0,5]
        ga0, kgh = soc_paras[:,2], soc_paras[:,6]
        muH = mu + kgh * ga0 - mu * kgh * ga0
        item1 = sigma * muH / ( sigma + muH ) / (1-ga0)
        P_degree = np.sum(P_matrix, axis=1) 
        item2 = 1/np.mean(P_degree)
        meanpA = torch.mean(pA,dim=0).cpu()
        weig = (1-(1-rA) * meanpA)
        item3 = 1/weig
        beta_threshold = item1 * item2 *item3
        return beta_threshold
