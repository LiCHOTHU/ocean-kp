import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np
from torch_geometric import utils
from torch_geometric.data import Data
from libs.layers import SAGPool


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.use_KNN = 5
        self.args = args
        self.num_features = args['num_features']
        self.nhid = args['nhid']
        self.num_classes = args['num_classes']
        self.pooling_ratio = args['pooling_ratio']
        self.dropout_ratio = args['dropout_ratio']

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, kp, label):
        data = self.build_graph_data(kp[0], label, self.use_KNN)

        x, edge_index = data.x, data.edge_index

        x_encode1 = F.relu(self.conv1(x, edge_index))
        x_encode1, x, edge_index, _, batch, perm = self.pool1(x_encode1, x, edge_index, None, None, None)
        x1 = torch.cat([gmp(x_encode1, batch), gap(x_encode1, batch)], dim=1)

        x_encode2 = F.relu(self.conv2(x_encode1, edge_index))
        x_encode2, x, edge_index, _, batch, perm = self.pool2(x_encode2, x, edge_index, None, batch, perm)
        x2 = torch.cat([gmp(x_encode2, batch), gap(x_encode2, batch)], dim=1)

        x_encode3 = F.relu(self.conv3(x_encode2, edge_index))
        x_encode3, x, edge_index, _, batch, perm = self.pool3(x_encode3, x, edge_index, None, batch, perm)
        x3 = torch.cat([gmp(x_encode3, batch), gap(x_encode3, batch)], dim=1)

        x_select = x
        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x, x_select, perm, edge_index

    def build_graph_data(self, kp, label, use_KNN):
        num_key = len(kp)
        if use_KNN != 0:
            adj = torch.from_numpy(np.identity(num_key).astype(np.int))
            select1 = torch.tensor([i for j in range(num_key - 1) for i in range(num_key)]).cuda()
            select2 = torch.tensor([(i % num_key) for j in range(1, num_key) for i in range(j, j + num_key)]).cuda()
            kp_select1 = torch.index_select(kp, 0, select1)
            kp_select2 = torch.index_select(kp, 0, select2)
            kp_dist = torch.norm((kp_select1 - kp_select2), dim=1).view(-1)
            for i in range(num_key):
                i_kp_dist = kp_dist[i::num_key]
                _, i_kp_near_idx = torch.topk(i_kp_dist, use_KNN, largest=False)
                kp_near_idx = (i_kp_near_idx + i + 1) % num_key
                adj[i, kp_near_idx] = 1
        else:
            adj = torch.from_numpy(np.ones(num_key, num_key).astype(np.int))

        edge_index = utils.dense_to_sparse(adj)
        edges = edge_index[0]
        x_axis = edges[0]
        y_axis = edges[1]
        edge_index = torch.stack([x_axis, y_axis], dim=0)
        edge_index = edge_index.clone().detach()
        x = kp.clone().detach()
        data = Data(x=x, edge_index=edge_index, y=label)
        return data.to(torch.cuda.current_device())
