import random
import numpy as np
import networkx as nx
import pickle
import os

def generate_graph(node_num, graph_type, seed = 123, ucm_gamma = 2.5, conk = 5.0):
    random.seed(seed)
    """生成随机网络。"""
    if graph_type == 'ba':
        m = 3  # 默认每个新节点连接的已有节点数为3
        G = nx.barabasi_albert_graph(node_num, m)

    elif graph_type == 'er':
        p = 6 / node_num  # 默认连接概率为6/n
        G = nx.erdos_renyi_graph(node_num, p)

    elif graph_type == 'ws':
        k = 6    # 默认每个节点的连接数为6
        p = 0.2  # 默认重新连接的概率为0.2
        G = nx.watts_strogatz_graph(node_num, k, p)

    elif graph_type == 'ucm':
        gamma = ucm_gamma
        degree_sequence = []
        while len(degree_sequence) < node_num:
            # 使用幂律分布生成随机数
            random_degree = int(conk*np.random.zipf(gamma))
            degree_sequence.append(random_degree)
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[0] += 1
        # 生成无相关配置模型
        G = nx.configuration_model(degree_sequence)
        # 转换为简单图以去除多边和自环
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))


    elif graph_type == 'ucm1':
        degree_upper_cutoff = int(np.sqrt(node_num))
        gamma = ucm_gamma
        degree_sequence = []
        while len(degree_sequence) < node_num:
            # 使用幂律分布生成随机数
            random_degree = int(conk*np.random.zipf(gamma))
            # 确保度值不超过上限
            if random_degree <= degree_upper_cutoff:
                degree_sequence.append(random_degree)
        if sum(degree_sequence) % 2 != 0:
            degree_sequence[0] += 1
        # 生成无相关配置模型
        G = nx.configuration_model(degree_sequence)
        # 转换为简单图以去除多边和自环
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

    elif graph_type == 'ucm2':
        # 计算刚性截止
        kc = int(node_num ** (1 / ucm_gamma))
        # 生成幂律度分布
        degrees = np.array(conk*np.random.zipf(ucm_gamma, node_num),dtype=int)
        degrees = np.minimum(degrees, kc)  # 应用刚性截止
        # 确保总度数为偶数
        if sum(degrees) % 2 != 0:
            degrees[np.argmax(degrees)] -= 1
        
        # 创建配置模型网络
        G = nx.configuration_model(degrees)
        
        # 转换为无自环和多重边的简单图
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
    
    else:
        raise ValueError("Unsupported graph type. Choose from 'ba', 'er' or 'ws'.")
    

    # 找到孤立点
    isolates = list(nx.isolates(G))
    for node in isolates:
        # 排除自身节点，从剩余节点中随机选择一个
        candidates = list(set(G.nodes()) - {node})
        if candidates:  # 确保有候选节点
            target = random.choice(candidates)
            G.add_edge(node, target)


    # 随机打乱图节点
    nodes = list(G.nodes)
    random.shuffle(nodes) 
    adj_matrix_csr = nx.to_scipy_sparse_array(G, format='csr')
    adj_matrix_relabeled = adj_matrix_csr[nodes, :]  # 重新排列行
    adj_matrix_relabeled = adj_matrix_relabeled[:, nodes]  # 重新排列列
    return adj_matrix_relabeled



def initialization(node_num, init_prob, method= 'prob', seed = 123):
    np.random.seed(seed)
    if method == 'prob': 
      node_features = np.tile(init_prob, (node_num, 1))

    elif method == 'state':
      node_features = np.zeros((node_num, len(init_prob)), dtype=float)
      for node in np.arange(node_num):
        index = np.random.choice(len(init_prob), p=init_prob)
        node_features[node,index] = 1.0
    return node_features

def epi_para(node_num, beta, sigma, mu, lam, delta, rA):
    return np.tile(np.array([beta, sigma, mu, lam, delta, rA]), (node_num, 1))
    




if __name__ == "__main__":
    # ---------------流行病参数---------------------------
    beta, sigma, mu = 0.3, 0.5, 0.1# S-->E,E-->I,I-->S
    lam, delta = 0.5, 0.3 #  U-->A, A-->U

    rA = 0.3 # discount factor

    xlm, xmh, gl, gm, gh = 1/3, 2/3, 0.1, 0.5, 0.9 # obs-->ga
    # xlm, xmh, gl, gm, gh = 1/2, 1/2, 0.05, 0.85, 0.9 # obs-->ga
    # xlm, xmh, gl, gm, gh = 1.0, 1.0, 0.05, 0.85, 0.9 # obs-->ga
    # xlm, xmh, gl, gm, gh = 0.99, 0.999, 0.05, 0.5, 0.95 # obs-->ga
    kgm, kgh = 0.5, 0.5 # ga-->ma, ga-->ha

    # ---------------初始分布参数-------------------------
    # SU, SA, EU, EA, IA
    init_prob = np.array([0.95, 0.0, 0.05, 0.0, 0.0])

    # ---------------拓扑连接参数--------------------------
    # node_num=1024
    node_num=5000
    Player, Ilayer= 'ucm1', 'ucm1' # inital_data_1.pkl
    # Player, Ilayer= 'ba', 'ba' #inital_data_2.pkl

    # ---------------------参数---------------------------
    init_data = {}
    epi_paras = epi_para(node_num, beta, sigma, mu, lam, delta, rA)
    init_data['epi_paras'] = epi_paras
    init_data['soc_paras'] = np.array([xlm, xmh, gl, gm, gh, kgm, kgh])

    init_data['P_matrix'] = generate_graph(node_num, Player, ucm_gamma = 2.5,\
                                            seed = 123, conk= 4.0)
    init_data['I_matrix'] = generate_graph(node_num, Ilayer, ucm_gamma = 2.5,\
                                            seed = 1234, conk= 5.0)
    # print(1.5 * 2**(1.5))
    # init_data['P_matrix'] = generate_graph(node_num, Player, seed = 12)
    # init_data['I_matrix'] = generate_graph(node_num, Ilayer, seed = 1)

    print(np.sum(init_data['P_matrix']))
    print(np.sum(init_data['I_matrix']))

    init_data['init_state'] = initialization(node_num, init_prob, method= 'state')
    init_data['init_prob'] = initialization(node_num, init_prob, method= 'prob')

    # # ---------------保存数据--------------------------
    # current_folder = os.path.dirname(os.path.abspath(__file__))
    # file_path = os.path.join(current_folder, 'inital_data_1.pkl')
    # with open(file_path, 'wb') as f:
    #     pickle.dump(init_data, f)