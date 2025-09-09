"""
GNN model for hardness regression of Shore A
The code presents 3 models:
1) GINEConv with edge_dim so edge_attr is used in message passing.
2) GCN
3) GAT
Select one model to run in the run_model.py script
"""

from typing import List
import torch
from torch import nn
from torch_geometric.nn import GINEConv, GINConv, GATv2Conv, GCNConv, global_mean_pool

def mlp(channels: List[int], act=nn.ReLU, dropout: float = 0.0):
    layers = []
    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i+1]))
        if i < len(channels) - 2:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class GINENet(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, use_edge_attr: bool, edge_dim: int, dropout: float):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self._uses_edge = []  # track which layers expect edge_attr

        for l in range(layers):
            nn_mlp = mlp([in_dim if l == 0 else hidden, hidden, hidden], dropout=dropout)

            if self.use_edge_attr and edge_dim > 0:
                self.convs.append(GINEConv(nn_mlp, edge_dim=edge_dim))
                self._uses_edge.append(True)
            else:
                # IMPORTANT: use GINConv (no edge attrs) instead of GINEConv
                self.convs.append(GINConv(nn_mlp))
                self._uses_edge.append(False)

        self.lin = mlp([hidden, hidden, 1], dropout=dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = x
        for conv, uses_edge in zip(self.convs, self._uses_edge):
            if uses_edge:
                if edge_attr is None:
                    raise RuntimeError("GINEConv layer requires edge_attr but got None. "
                                       "Either enable --use-edge-attr and provide edge features, "
                                       "or construct the model without edge features.")
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = torch.relu(h)
            if self.dropout > 0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.lin(g).view(-1)

class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for l in range(layers):
            self.convs.append(GCNConv(in_dim if l == 0 else hidden, hidden))
        self.lin = mlp([hidden, hidden, 1], dropout=dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = x
        for conv in self.convs:
            h = torch.relu(conv(h, edge_index))
            if self.dropout > 0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.lin(g).view(-1)


class GATNet(nn.Module):
    def __init__(self, in_dim, hidden, layers, heads=4, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for l in range(layers):
            in_ch = in_dim if l == 0 else hidden * heads
            self.convs.append(GATv2Conv(in_ch, hidden, heads=heads, dropout=dropout, concat=True))
        self.lin = mlp([hidden*heads, hidden, 1], dropout=dropout)

    def forward(self, x, edge_index, batch, edge_attr=None):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = torch.relu(h)
            if self.dropout > 0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        return self.lin(g).view(-1)