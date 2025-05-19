import numpy as np
import torch
from torch_scatter import scatter_mean
import os,sys
sys.path.append(os.getcwd())
from methods.utils import update_social_attention, graph_dynamic

from torch_geometric.data import Data
from torch_geometric.utils import scatter
from torch_geometric.nn import MessagePassing
import time 


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

class MMCA():
    def __init__(self, device, delta_t = 0.1):
      self.delta_t = delta_t
      self.device = device
      self.MessagePassing_model = NeighborFeatureWithEdgeMultiplication()

    def update_epidemic_parameters(self, features_prob_tensor, epi_paras, soc_paras,\
                                    P_edge_index, I_edge_index, communities):
        beta, sigma, mu, lam, delta, rA = epi_paras[...,0] * self.delta_t,\
                        epi_paras[...,1] * self.delta_t, epi_paras[...,2] * self.delta_t,\
                        epi_paras[...,3] * self.delta_t, epi_paras[...,4] * self.delta_t, epi_paras[...,5]
        

        communities = torch.tensor(communities, device=self.device)
        statEI = torch.sum(features_prob_tensor[:,:,[2,3,4]],dim=2).T

        obs = scatter_mean(statEI, communities, dim=0) 
        ga, ma, ha = update_social_attention(obs, soc_paras)

        ma, ha = self.delta_t * ma, self.delta_t *ha

        lamM = ma[communities]
        muH = 1-(1-mu)*(1-ha[communities].T)
        beta_G = beta * (1 - ga[communities].T)
        beta_GA = beta_G * rA
        statA = torch.sum(features_prob_tensor[:,:,[1,3,4]],dim=2).T
        edge_attr_EI_G = beta_G[:,P_edge_index[1]].T
        edge_attr_EI_GA = beta_GA[:,P_edge_index[1]].T
        edge_attr_A = lam[:,I_edge_index[1]].T
        r = ((1 - lamM) * self.MessagePassing_model(statA, I_edge_index, edge_attr_A)).T
        qU = self.MessagePassing_model(statEI, P_edge_index, edge_attr_EI_G).T
        qA = self.MessagePassing_model(statEI, P_edge_index, edge_attr_EI_GA).T
        return r, qA, qU, delta, sigma, muH
    
    def update(self, features_prob_tensor, epi_paras, soc_paras, P_edge_index, I_edge_index, communities, obs = None):
        r, qA, qU, delta, sigma, muH = self.update_epidemic_parameters(features_prob_tensor, epi_paras,\
                                                                      soc_paras, P_edge_index, I_edge_index, communities)
        features = features_prob_tensor  # 形状: (128, 10000, 5)
        # 更新特征
        update_features = torch.zeros_like(features)  # 形状: (128, 10000, 5)
        update_features[..., 0] = r * qU * features[..., 0] + delta * qU * features[..., 1] + delta * muH * features[..., 4]
        update_features[..., 1] = (1.0 - r) * qA * features[..., 0] + (1.0 - delta) * qA * features[..., 1] + (1.0 - delta) * muH * features[..., 4]
        update_features[..., 2] = r * (1.0 - qU) * features[..., 0] + delta * (1.0 - qU) * features[..., 1] + r * (1.0 - sigma) * features[..., 2] + delta * (1.0 - sigma) * features[..., 3]
        update_features[..., 3] = (1.0 - r) * (1.0 - qA) * features[..., 0] + (1.0 - delta) * (1.0 - qA) * features[..., 1] + (1.0 - r) * (1.0 - sigma) * features[..., 2] + (1.0 - delta) * (1.0 - sigma) * features[..., 3]
        update_features[..., 4] = sigma * features[..., 3] + sigma * features[..., 2] + (1.0 - muH) * features[..., 4]

        return update_features


if __name__ == "__main__":
    # ---------------初始数据-----------------------------
    import os
    import pickle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(current_folder, 'parameters\community_data_1.pkl')
    with open(file_path, 'rb') as f:
        init_data = pickle.load(f)

    epi_paras = torch.tensor(init_data['epi_paras'], dtype=torch.float32).to(device)
    soc_paras = torch.tensor(init_data['soc_paras'], dtype=torch.float32).to(device)
    features_prob = torch.tensor(init_data['init_state'], dtype=torch.float32).to(device)
    P_rows, P_cols = init_data['P_matrix'].nonzero()
    P_edge_index = torch.tensor(np.array([P_rows, P_cols]), dtype=torch.long).to(device)
    communities = init_data['communities']
    I_rows, I_cols = init_data['I_matrix'].nonzero()
    I_edge_index = torch.tensor(np.array([I_rows, I_cols]), dtype=torch.long).to(device)

    para_len = 1
    node_num =  features_prob.shape[0]

    soc_paras = soc_paras.unsqueeze(0).repeat(para_len, 1)
    epi_paras = epi_paras.unsqueeze(0).repeat(para_len, 1, 1)

    features_prob_tensor = features_prob.unsqueeze(0).repeat(para_len, 1, 1).to(device)


    mmca = MMCA(device)
    time_scale = 1000
    start = time.time()
    features_times, _ = graph_dynamic(time_scale, mmca, features_prob_tensor, epi_paras, soc_paras,\
                            P_edge_index, I_edge_index, communities, device)
    print(features_times.shape)
    print(time.time()-start)
