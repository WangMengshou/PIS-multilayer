import torch
from torch_scatter import scatter_mean

# 政府的反馈函数
def government_feedback(obs, xlm = 1/3, xmh = 2/3, gl = 0.05, gm = 0.5, gh = 0.95):
    obs = torch.clamp(obs, 0, 1)
    response = torch.ones_like(obs) * gm 
    response = torch.where(obs < xlm, gl, response)
    response = torch.where(obs > xmh, gh, response)
    return response

def media_hospital_feedback(g, k=0.5):
    g = torch.clamp(g, 0, 1)
    return torch.clamp(k * g, 0, 1)

def update_social_attention(obs, soc_paras):
    xlm, xmh, gl, gm, gh, kgm, kgh =\
        soc_paras[:,0],soc_paras[:,1],soc_paras[:,2],soc_paras[:,3],\
        soc_paras[:,4],soc_paras[:,5],soc_paras[:,6],
    ga = government_feedback(obs, xlm, xmh, gl, gm, gh) # goverment attention
    ma = media_hospital_feedback(ga, kgm) # media attention
    ha = media_hospital_feedback(ga, kgh) # hospital attention
    return ga, ma, ha

#^^^^^^^^^sigmal^^^^^^^^^^^media and hospital attention

def normalized_sigmoid(x, k=10, center=0.5, scale = 0.8, x0 = 0.1):
    k = torch.tensor(k, dtype=x.dtype, device=x.device)
    center = torch.tensor(center, dtype=x.dtype, device=x.device)
    x0 = torch.tensor(x0, dtype=x.dtype, device=x.device)
    x = torch.clamp(x, 0, 1)  # Ensure x is within [0, 1]
    # Calculate sigmoid
    sigmoid = 1 / (1 + torch.exp(-k * (x - center)))
    min_sigmoid = 1 / (1 + torch.exp(-k * (0 - center)))
    max_sigmoid = 1 / (1 + torch.exp(-k * (1 - center)))
    return x0 + scale * (sigmoid - min_sigmoid) / (max_sigmoid - min_sigmoid)

def update_social_attention_sigmoid(obs, soc_paras, sg, sgm, sgh):
    xlm, xmh, gl, gm, gh, kgm, kgh =\
        soc_paras[:,0],soc_paras[:,1],soc_paras[:,2],soc_paras[:,3],\
        soc_paras[:,4],soc_paras[:,5],soc_paras[:,6],
    ga = normalized_sigmoid(obs, k=sg, center=0.5 * (xlm + xmh), scale = gh-gl, x0 = gl) # goverment attention
    ma = normalized_sigmoid(ga, k=sgm, center=0.5, scale = kgm, x0 = 0) # media attention
    ha = normalized_sigmoid(ga, k=sgh, center=0.5, scale = kgh, x0 = 0) # hospital attention
    return ga, ma, ha

# 时间演化
def dynamic(time_scale, method, features, epi_paras, soc_paras, P_edge_index, I_edge_index, device, obs = 1):
    features_times = torch.mean(features,dim=1).clone().detach().unsqueeze(1).to(device)
    for i in torch.arange(time_scale):
      features = method.update(features, epi_paras, soc_paras, P_edge_index, I_edge_index, obs)
      features_mean = torch.mean(features, dim=1)
      features_times = torch.cat((features_times, features_mean.clone().detach().unsqueeze(1).to(device)), dim=1)
      
      if (i+1)%100==0:
        print(f"time:{i+1}", end='\r')
    return features_times.float().to('cpu'), features


# 计算动态敏感性峰值
def calculate_susceptibility(rho,N):
    N = torch.tensor(N, dtype=torch.float32)
    rho_mean = torch.mean(rho, dim=1)
    rho_sq_mean = torch.mean(rho**2, dim=1)
    chi = torch.sqrt(N) * (rho_sq_mean - rho_mean**2) / rho_mean
    return chi, torch.sqrt(N) * rho_mean


# 时间演化
def graph_dynamic(time_scale, method, features, epi_paras, soc_paras, P_edge_index, I_edge_index, communities, device, obs = 1):
    features_times = torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)
    for i in torch.arange(time_scale):
      features = method.update(features, epi_paras, soc_paras, P_edge_index, I_edge_index, communities, obs)
      if (i+1)%10==0:
        features_times = torch.cat((features_times, torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)), dim=0)
      if (i+1)%100==0:
        print(f"time:{i+1}", end='\r')
    return features_times.float().to('cpu'), features


def graph_dynamic_delays(time_scale, method, features, epi_paras, soc_paras, P_edge_index, I_edge_index, communities, device, delays, obs = 1, sigmoid = None, soc_attention_constant = None):
    communities = torch.tensor(communities, device=device)
    features_times = torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)
    soc_attention = torch.zeros((time_scale+delays, 3, max(communities)+1, features.shape[0]), device=device)
    for i in torch.arange(time_scale):
      if soc_attention_constant is not None:
        soc_attention = soc_attention_constant.to(device)
      else:
        statEI = torch.sum(features[:,:,[2,3,4]],dim=2).T
        obs_delay = scatter_mean(statEI, communities, dim=0) 
        if sigmoid != None:
          sg, sgm, sgh = sigmoid[0], sigmoid[1], sigmoid[2]
          ga, ma, ha = update_social_attention_sigmoid(obs_delay, soc_paras, sg, sgm, sgh)
          ga, ma, ha = ga, ma + 0.05, ha +0.05
          # ga, ma, ha = ga, 0*ma, ha
          # ga, ma, ha = ga, ma, 0*ha
          # ga, ma, ha = ga, 0*ma, 0*ha

        else:
          ga, ma, ha = update_social_attention(obs_delay, soc_paras)
          ga, ma, ha = ga, ma + 0.05, ha +0.05 # 这里是与sigmoid反馈时有相同的初始值，考虑delay测试时，这个注释掉。
        soc_attention[i+delays, 0] = ga
        soc_attention[i+delays, 1] = ma
        soc_attention[i+delays, 2] = ha

      features = method.update(features, epi_paras, soc_attention[i], P_edge_index, I_edge_index, communities, obs)
      if (i+1)%10==0:
        features_times = torch.cat((features_times, torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)), dim=0)
      if (i+1)%100==0:
        print(f"time:{i+1}", end='\r')
    return features_times.float().to('cpu'), features, soc_attention.clone().detach().to('cpu')