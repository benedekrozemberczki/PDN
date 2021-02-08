import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AttentionNet(torch.nn.Module):
    def __init__(self, node_features, edge_features, node_filters, edge_filters, classes):
        super(AttentionNet, self).__init__()
        self.dense_1 = torch.nn.Linear(edge_features, edge_filters)
        self.dense_2 = torch.nn.Linear(edge_filters, 1)
        self.convolution_1 = GCNConv(node_features, node_filters)
        self.convolution_2 = GCNConv(node_filters, classes)

    def forward(self, x, edge_index, edge_x):
        edge_x = F.relu(self.dense_1(edge_x))
        edge_x = torch.sigmoid(self.dense_2(edge_x)).view(-1)
        x = self.convolution_1(x, edge_index, edge_x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.convolution_2(x, edge_index, edge_x)
        return F.log_softmax(x, dim=1)
