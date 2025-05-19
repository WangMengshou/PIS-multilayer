import torch

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
def graph_dynamic(time_scale, method, features, epi_paras, soc_paras, P_edge_index, I_edge_index, communities,
                   device, obs = 1):
    features_times = torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)
    for i in torch.arange(time_scale):
      features = method.update(features, epi_paras, soc_paras, P_edge_index, I_edge_index, communities, obs)
      if (i+1)%10==0:
        features_times = torch.cat((features_times, torch.mean(features, dim=0).clone().detach().unsqueeze(0).to(device)), dim=0)
      if (i+1)%100==0:
        print(f"time:{i+1}", end='\r')
    return features_times.float().to('cpu'), features