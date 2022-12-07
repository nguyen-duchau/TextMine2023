import numpy as np 
import torch as th 
import torch.nn.functional as F
import torch.nn as nn 
from torch_geometric.nn import GCNConv, global_mean_pool

import operator as op
import itertools as it, functools as ft 

class PositionalEncoding(nn.Module):
    def __init__(self, max_nb_nodes, in_dim, drop_val=0.1):
        super(PositionalEncoding, self).__init__()
        pos = np.arange(0, max_nb_nodes)[:, None]
        idx = np.fromfunction(lambda _,j: j - j % 2, shape=(1, in_dim))
        mask = np.fromfunction(lambda _,j: j % 2 == 0, shape=(1, in_dim))
        pnt = pos / (10000 ** (idx / in_dim))
        val = np.sin(pnt) * mask + np.cos(pnt) * (1 - mask)
        self.drop_layer = nn.Dropout(drop_val)
        self.register_buffer('psne_layer', th.tensor(val).float())

    def forward(self, node_positions):
        pos = self.psne_layer[node_positions, :]
        return self.drop_layer(pos)

class NRLModel(nn.Module):
    def __init__(self):
        super(NRLModel, self).__init__()
        self.activation_map = {
            0: nn.Identity(), 
            1: nn.ReLU(), 
            2: nn.LeakyReLU(0.2),
            3: nn.GELU(), 
            4: nn.Tanh(), 
            5: nn.Sigmoid(), 
            6: nn.Softmax(dim=-1)
        }

class MLPModel(NRLModel):
    def __init__(self, layers_config, activations, apply_norm=False):
        super(MLPModel, self).__init__()

        assert isinstance(layers_config, list)
        assert isinstance(activations, list)
        assert len(layers_config)  - 1 == len(activations) 

        self.shapes = list(zip(layers_config[:-1], layers_config[1:]))
        self.activations = activations 
        self.linear_layers = nn.ModuleList([]) 
        
        for index, (in_dim, out_dim) in enumerate(self.shapes):
            linear = nn.Linear(in_features=in_dim, out_features=out_dim)
            nonlinear = self.activation_map[self.activations[index]] 
            if apply_norm and index < len(self.shapes) - 1:
                normalizer = nn.BatchNorm1d(out_dim)
                layer = nn.Sequential(linear, normalizer, nonlinear)
            else:
                layer = nn.Sequential(linear, nonlinear)
            self.linear_layers.append(layer)
    
    def forward(self, X_0):
        X_N = ft.reduce(lambda X_I, LIN: LIN(X_I), self.linear_layers, X_0)
        return X_N 
    

# class GCNModel(NRLModel):
#     def __init__(self, layers_config, activations, drop_val=0.1, acc='mean'):
#         super(GCNModel, self).__init__()

#         assert isinstance(layers_config, list)
#         assert isinstance(activations, list)
#         assert len(layers_config) - 1 == len(activations) 

#         self.acc = acc 
#         self.shapes = list(zip(layers_config[:-1], layers_config[1:]))
#         self.activations = activations 
#         self.convolutions = nn.ModuleList([])
        
#         for index, (in_dim, out_dim) in enumerate(self.shapes):
#             fun = self.activation_map.get(self.activations[index], nn.Identity()) 
#             layer = dglnn.SAGEConv(
#                 in_feats=in_dim, 
#                 out_feats=out_dim, 
#                 aggregator_type=self.acc,
#                 activation=fun,
#                 feat_drop=drop_val
#             )
#             self.convolutions.append(layer)
    
#     def forward(self, G, X_0, E_0):
#         X_N = ft.reduce(
#             lambda X_I, GCN: GCN(G, X_I, E_0), 
#             self.convolutions, 
#             X_0 
#         )
#         with G.local_scope():
#             G.ndata['X_N'] = X_N 
#             graph_embedding = dgl.readout_nodes(G, 'X_N')
#             return graph_embedding

# class GNNModel(nn.Module):
#     def __init__(self, config):
#         super(GNNModel, self).__init__()
#         self.fst_gcn_vectorizer = GCNModel(**config)
#         self.snd_gcn_vectorizer = GCNModel(**config)
    
#     def forward(self, GF, GS, XF_0, XS_0, EF_0=None, ES_0=None):
#         XF_N = self.fst_gcn_vectorizer(GF, XF_0, E_0=EF_0)
#         XS_N = self.snd_gcn_vectorizer(GS, XS_0, E_0=ES_0)
#         return XF_N, XS_N

#     def graph_embedding(self, G, X_0, E_0):
#         return self.fst_gcn_vectorizer(G, X_0, E_0).cpu().numpy()

class SimpleGCN(th.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 756)
        self.conv2 = GCNConv(756, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.embedding, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return global_mean_pool(x, data.batch)