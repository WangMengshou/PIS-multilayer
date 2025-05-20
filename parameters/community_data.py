import networkx as nx
import random
import numpy as np
import os
import pickle

from Initial_data import epi_para, generate_graph, initialization

def generate_community_graph(k, m=1, seed=None):
    """
    生成社区图（BA 网络），确保没有孤立节点。
    参数:
    - k: 社区数量（节点数）
    - m: 每个新节点连接的边数
    返回:
    - community_graph: 生成的社区图
    """
    # 生成 BA 网络
    if k == 1:
        community_graph = nx.Graph()
        community_graph.add_node(0)
    else:
        community_graph = nx.barabasi_albert_graph(k, m, seed=seed)
        
        # 确保没有孤立节点
        while not nx.is_connected(community_graph):
            # 如果网络不连通，重新生成
            seed += 1
            community_graph = nx.barabasi_albert_graph(k, m, seed=seed)
    
    return community_graph

def generate_node_based_graph(n, m, p_in, community_graph, seed=None):
    """
    基于社区图生成节点图。

    参数:
    - n: 节点数量
    - m: 每个新节点连接的边数
    - p_in: 同一社区内连接的概率
    - community_graph: 社区图

    返回:
    - G: 生成的节点图
    - communities: 节点的社区标签
    """
    # 固定随机种子
    if seed is not None:
        random.seed(seed)

    # 初始化图
    k = len(community_graph)
    G = nx.Graph()
    
    # 为每个节点分配社区标签
    communities = [i % k for i in range(n)]
    if k == 1:
        G = nx.barabasi_albert_graph(n, m)
    else:
        # 初始化前 k 个节点，确保每个社区至少有一个节点
        for i in range(m*k):
            G.add_node(i)
            communities[i] = i%k  # 每个社区至少有一个节点
        
        # 添加剩余节点
        for i in range(m*k, n):
            G.add_node(i)
            communities[i] = random.randint(0, k - 1)
            
            # 选择 m 个邻居
            neighbors = []
            while len(neighbors) < m:
                # 根据概率选择连接方式
                if random.random() < p_in:
                    # 优先选择同一社区内的节点
                    candidates = [node for node in G.nodes if communities[node] == communities[i]]
                else:
                    # 选择其他社区的节点
                    # 根据社区图的邻居选择目标社区
                    connected_communities = list(community_graph.neighbors(communities[i]))
                    # 随机选择一个连接的社区
                    target_community = random.choice(connected_communities)
                    # 在目标社区内选择节点
                    candidates = [node for node in G.nodes if communities[node] == target_community]
                # 基于节点度来选择
                degrees = [G.degree(node) for node in candidates]
                total_degree = sum(degrees)
                if total_degree == 0:
                    candidate = random.choice(candidates)
                else:
                    probabilities = [degree / total_degree for degree in degrees]
                    candidate = random.choices(candidates, weights=probabilities, k=1)[0]
                
                # 确保没有自环和重复邻居
                if candidate != i and candidate not in neighbors:
                    neighbors.append(candidate)
            
            # 添加边
            for neighbor in neighbors:
                G.add_edge(i, neighbor)
    
    return G, communities

if __name__ == "__main__":
    # ---------------流行病参数---------------------------
    beta, sigma, mu = 0.3, 0.5, 0.1# S-->E,E-->I,I-->S
    lam, delta = 0.5, 0.3 #  U-->A, A-->U

    rA = 0.3 # discount factor

    xlm, xmh, gl, gm, gh = 1/3, 2/3, 0.1, 0.5, 0.9 # obs-->ga
    kgm, kgh = 0.5, 0.5 # ga-->ma, ga-->ha

    # ---------------初始分布参数-------------------------
    # SU, SA, EU, EA, IA
    init_prob = np.array([0.99, 0.0, 0.01, 0.0, 0.0])

    # ---------------拓扑连接参数--------------------------
    node_num=5000
    Player, Ilayer= 'ba', 'ba'

    # ---------------------参数---------------------------
    init_data = {}
    epi_paras = epi_para(node_num, beta, sigma, mu, lam, delta, rA)
    init_data['epi_paras'] = epi_paras
    init_data['soc_paras'] = np.array([xlm, xmh, gl, gm, gh, kgm, kgh])

    init_data['P_matrix'] = generate_graph(node_num, Player, seed = 12)


    community_num = 10
    P_community = generate_community_graph(k = community_num, m=2, seed=13)
    P_G, communities = generate_node_based_graph(node_num, m = 3, p_in = 0.9,\
                                                  community_graph = P_community, seed=0)
    init_data['P_community'] = P_community
    init_data['P_G'] = P_G
    init_data['communities'] = communities

    init_data['P_matrix'] = adj_matrix_csr = nx.to_scipy_sparse_array(P_G, format='csr')
    init_data['I_matrix'] = generate_graph(node_num, Ilayer, seed = 1)

    print(np.sum(init_data['P_matrix']))
    print(np.sum(init_data['I_matrix']))

    # 初始化
    community_infect = np.where(np.array(communities) == 0)[0].tolist()
    init_prob = np.array([0.95, 0.0, 0.05, 0.0, 0.0])
    node_features = np.zeros((node_num, len(init_prob)), dtype=float)
    node_features[:,0] = 1.0
    node_features[community_infect, 0] = 0.0
    for node in community_infect:
        index = np.random.choice(len(init_prob), p=init_prob)
        node_features[node,index] = 1.0
    init_data['init_state'] = node_features
    
    # ---------------保存数据--------------------------
    current_folder = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_folder, 'community_data_1.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(init_data, f)