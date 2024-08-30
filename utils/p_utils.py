import torch
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import math
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity(graph):
    """
       计算图中有边相连的节点之间的相似度并返回作为对应边的特征。

       参数:
       graph: NetworkX图对象
       node_features: 一个字典，键是节点，值是节点特征的列表或数组

       返回:
       一个字典，键是边，值是计算出的相似度特征
       """
    similarity = []

    node_features = graph.h.numpy()

    # 遍历图中的每条边
    for u, v in graph.edges.numpy():
        # 计算两个节点特征的余弦相似度
        feature_u = node_features[u]
        feature_v = node_features[v]
        _ = cosine_similarity([feature_u], [feature_v])[0][0]

        # 将计算出的相似度作为边的特征存储
        similarity.append(_)

    return similarity


def compute_accuracy(model, graphs, device, batch_size=64):
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), batch_size):
        curr_idx = idx[i: i + batch_size]
        if len(curr_idx) == 0:
            continue
        output.append(model([graphs[j] for j in curr_idx]).detach())
    output = torch.cat(output, 0)

    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(graphs))

    return acc


def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


# a = preprocess_input([0, 0, 0, 1], [6,7,6,6],2,, 'cpu')
