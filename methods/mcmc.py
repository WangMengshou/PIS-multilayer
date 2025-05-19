import numpy as np
import torch
import os, sys
sys.path.append(os.getcwd())
from methods.utils import update_social_attention, dynamic, calculate_susceptibility

from torch_geometric.data import Data
from torch_geometric.utils import scatter
import time 

class MCMC():
    def __init__(self, para_len, node_num, device, delta_t = 0.1):
       self.para_len = para_len
       self.M = 50
       self.Prep = 0.1
       self.device = device
       self.delta_t = delta_t
       self.active_configurations =  torch.zeros(self.para_len, self.M, node_num, 5, device=self.device)

    def update_epidemic_parameters(self, features_state_tensor, epi_paras, soc_paras, P_edge_index, I_edge_index):
        beta, sigma, mu, lam, delta, rA = epi_paras[...,0] * self.delta_t,\
                        epi_paras[...,1] * self.delta_t, epi_paras[...,2] * self.delta_t,\
                        epi_paras[...,3] * self.delta_t, epi_paras[...,4] * self.delta_t, epi_paras[...,5]
        
        statEI = torch.sum(features_state_tensor[:,:,[2,3,4]],dim=2).T
        obs = torch.mean(statEI, dim=0)
        ga, ma, ha = update_social_attention(obs, soc_paras)
        ma, ha = self.delta_t * ma, self.delta_t *ha
        # 邻居的感染数目
        data_EI = Data(x=statEI, edge_index=P_edge_index)
        data_A = Data(x=torch.sum(features_state_tensor[:,:,[1,3,4]],dim=2).T, edge_index=I_edge_index)

        count_EI = scatter(data_EI.x[data_EI.edge_index[1]],  # 目标节点的特征
                            data_EI.edge_index[0],          # 源节点索引
                            dim=0,                       # 沿着节点维度聚合
                            reduce='sum').T                # 聚合方式为求和
        count_A = scatter(data_A.x[data_A.edge_index[1]],  # 目标节点的特征
                            data_A.edge_index[0],          # 源节点索引
                            dim=0,                       # 沿着节点维度聚合
                            reduce='sum').T                # 聚合方式为求和
        lamM = ma
        muH = 1-(1-mu)*(1-ha).unsqueeze(1)
        beta_G = beta * (1 - ga).unsqueeze(1)
        beta_GA = beta_G * rA
        r = (1 - lamM).unsqueeze(1) * (1 - lam) ** count_A
        qA = (1 - beta_GA) ** count_EI
        qU = (1 - beta_G) ** count_EI
        return r, qA, qU, delta, sigma, muH 

    def update_features(self, features_state_tensor, r, qA, qU, delta, sigma, muH):

        # 探索阈值时候使用这个
        randI = torch.rand(r.shape[1], device=self.device).unsqueeze(0)
        randPA = torch.rand(qA.shape[1], device=self.device).unsqueeze(0)
        randPU = torch.rand(qU.shape[1], device=self.device).unsqueeze(0)

        # # 跑多条轨迹时候使用这个
        # randI = torch.rand(r.shape, device=self.device)
        # randPA = torch.rand(qA.shape, device=self.device)
        # randPU = torch.rand(qU.shape, device=self.device)

        c0 = features_state_tensor[:,:,0]
        c1 = features_state_tensor[:,:,1]
        c2 = features_state_tensor[:,:,2]
        c3 = features_state_tensor[:,:,3]
        c4 = features_state_tensor[:,:,4]

        c01 = c0 + (randI>=r) + (randPA >= qA) == 3
        c02 = c0 + (randI>=r) + (randPA < qA) == 3
        c03 = c0 + (randI<r) + (randPU >= qU) == 3

        c11 = c1 + (randI>=delta) + (randPA >= qA) == 3
        c12 = c1 + (randI<delta) + (randPU >= qU) == 3
        c13 = c1 + (randI<delta) + (randPU < qU) == 3

        c21 = c2 + (randI>=r) + (randPA >= sigma) == 3
        c22 = c2 + (randI>=r) + (randPA < sigma) == 3
        c23 = c2 + (randI<r) + (randPU < sigma) == 3

        c31 = c3 + (randI>=delta) + (randPA < sigma) == 3
        c32 = c3 + (randI<delta) + (randPU >= sigma) == 3
        c33 = c3 + (randI<delta) + (randPU < sigma) == 3

        c41 = c4 + (randI>=delta) + (randPA < muH) == 3
        c42 = c4 + (randI<delta) + (randPU < muH) == 3

        features_state_tensor[:,:,0][c01], features_state_tensor[:,:,3][c01] = 0, 1
        features_state_tensor[:,:,0][c02], features_state_tensor[:,:,1][c02] = 0, 1
        features_state_tensor[:,:,0][c03], features_state_tensor[:,:,2][c03] = 0, 1

        features_state_tensor[:,:,1][c11], features_state_tensor[:,:,3][c11] = 0, 1
        features_state_tensor[:,:,1][c12], features_state_tensor[:,:,2][c12] = 0, 1
        features_state_tensor[:,:,1][c13], features_state_tensor[:,:,0][c13] = 0, 1


        features_state_tensor[:,:,2][c21], features_state_tensor[:,:,3][c21] = 0, 1
        features_state_tensor[:,:,2][c22], features_state_tensor[:,:,4][c22] = 0, 1
        features_state_tensor[:,:,2][c23], features_state_tensor[:,:,4][c23] = 0, 1


        features_state_tensor[:,:,3][c31], features_state_tensor[:,:,4][c31] = 0, 1
        features_state_tensor[:,:,3][c32], features_state_tensor[:,:,2][c32] = 0, 1
        features_state_tensor[:,:,3][c33], features_state_tensor[:,:,4][c33] = 0, 1

        features_state_tensor[:,:,4][c41], features_state_tensor[:,:,1][c41] = 0, 1
        features_state_tensor[:,:,4][c42], features_state_tensor[:,:,0][c42] = 0, 1      

        return features_state_tensor
      
    def update(self, features_state_tensor, epi_paras, soc_paras, P_edge_index, I_edge_index, obs = None):
      r, qA, qU, delta, sigma, muH = self.update_epidemic_parameters(features_state_tensor, epi_paras,\
                                                                      soc_paras, P_edge_index, I_edge_index)
      update_features_tensor = self.update_features(features_state_tensor, r, qA, qU, delta, sigma, muH)
      return update_features_tensor

    def update_QS(self, features_state_tensor, epi_paras, soc_paras, P_edge_index, I_edge_index, t_max):
      t = 0
      rho = torch.zeros(features_state_tensor.shape[0], t_max+1, device=self.device)
      features_tensor  = features_state_tensor.clone().to(self.device)
      active_configurations_index = torch.ones(self.para_len, dtype=torch.long, device=self.device)
      self.active_configurations[:,0,:,:] = features_state_tensor.clone().to(self.device)
      rho[:,t] = torch.mean(torch.sum(features_tensor[:,:,[2,3,4]],dim = 2),dim=1)

      while t < t_max: 
        # 替换掉感染人数为零的状态
        update_state_index = (torch.sum(features_tensor[:,:,[2,3,4]], dim=[1,2]) == 0)
        random_int = (torch.rand_like(active_configurations_index[update_state_index], dtype=torch.float32)\
                          * (active_configurations_index[update_state_index])).to(torch.int64)
        features_tensor[update_state_index,:,:] = self.active_configurations[update_state_index,random_int,:,:]
        #演化动力学
        r, qA, qU, delta, sigma, muH = self.update_epidemic_parameters(features_tensor, epi_paras,\
                                                                      soc_paras, P_edge_index, I_edge_index)
        features_tensor = self.update_features(features_state_tensor, r, qA, qU, delta, sigma, muH)
        t+=1
        rho[:,t] = torch.mean(torch.sum(features_tensor[:,:,[2,3,4]],dim = 2),dim=1)

        # 更新配置状态集合
        if torch.rand(1).item() < self.Prep:
            update_configurations_element = (torch.sum(features_tensor[:,:,[2,3,4]], dim=[1,2]) != 0)
            update_configurations_index = active_configurations_index[update_configurations_element]
            update_configurations_index[update_configurations_index == self.M] =\
              torch.randint(0, self.M, size=update_configurations_index[update_configurations_index == self.M].shape, device=self.device)
            self.active_configurations[update_configurations_element,update_configurations_index,:,:] \
                = features_tensor[update_configurations_element]
            active_configurations_index[update_configurations_element & (active_configurations_index < self.M)] += 1
        if t%100 ==0:
            print(f'time:{t}', end='\r')
        # if t > 0.2 * t_max:
        #     self.Prep = 0.01

      return rho.cpu()

if __name__ == "__main__":
  # ---------------初始数据-----------------------------
  import os
  import pickle
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  current_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  file_path = os.path.join(current_folder, 'parameters\inital_data_1.pkl')
  with open(file_path, 'rb') as f:
      init_data = pickle.load(f)

  epi_paras = torch.tensor(init_data['epi_paras'], dtype=torch.float32).to(device)
  soc_paras = torch.tensor(init_data['soc_paras'], dtype=torch.float32).to(device)
  features_state = torch.tensor(init_data['init_state'], dtype=torch.float32).to(device)
  P_rows, P_cols = init_data['P_matrix'].nonzero()
  P_edge_index = torch.tensor(np.array([P_rows, P_cols]), dtype=torch.long).to(device)
  I_rows, I_cols = init_data['I_matrix'].nonzero()
  I_edge_index = torch.tensor(np.array([I_rows, I_cols]), dtype=torch.long).to(device)


  para_len = 16
  node_num = features_state.shape[0]
  soc_paras = soc_paras.unsqueeze(0).repeat(para_len, 1)
  epi_paras = epi_paras.unsqueeze(0).repeat(para_len, 1, 1)
  # epi_paras[...,[0,1,2,3,4]] = epi_paras[...,[0,1,2,3,4]] * delta_t
  # print(epi_paras.shape)
  
#   # 替换掉自己的参数
#   beta_log = torch.linspace(-1, 0, para_len)
#   beta_values = 10 ** beta_log
#   beta = beta_values.unsqueeze(1).expand(-1, node_num).to(device)
#   epi_paras[...,0] = beta


  features_state_tensor = features_state.unsqueeze(0).repeat(para_len, 1, 1).to(device)
  mcmc = MCMC(para_len, node_num, device)
  time_scale = 1000
  start = time.time()
  features_times, _ = dynamic(time_scale, mcmc, features_state_tensor, epi_paras, soc_paras,\
                               P_edge_index, I_edge_index, device)
  print(torch.mean(features_times,dim=0))
#   print(features_times[1,:,:])
  print(time.time()-start)


  # # # # ---------------初始数据-----------------------------

  # para_len = 64
  # node_num =  features_state.shape[0]
  # soc_paras = soc_paras.unsqueeze(0).repeat(para_len, 1)
  # epi_paras = epi_paras.unsqueeze(0).repeat(para_len, 1, 1) 
  
  # # 替换掉自己的参数
  # beta_log = torch.linspace(-2.5, -1.5, para_len)
  # beta_values = 10 ** beta_log
  # beta = beta_values.unsqueeze(1).expand(-1, node_num).to(device)
  # epi_paras[...,0] = beta



  # features_state_tensor = features_state.unsqueeze(0).repeat(para_len, 1, 1).to(device)
  # mcmc = MCMC(para_len, node_num, device)
  # t_max = 50000
  # t_ave = int(t_max*0.8)

  # start = time.time()
  # rho = mcmc.update_QS(features_state_tensor, epi_paras, soc_paras, P_edge_index, I_edge_index, t_max)
  # chis, rho_means = calculate_susceptibility(rho[:,-t_ave:] ,node_num)

  # print(chis)
  # print(rho_means)
  # print(time.time()-start)


  # torch.save(chis.cpu(), 'methods/data/chi_values.pt')
  # torch.save(rho_means.cpu(), 'methods/data/rho_mean_values.pt')
  # torch.save(beta_log.cpu(), 'methods/data/beta_values.pt')
  
