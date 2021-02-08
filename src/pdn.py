import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PathfinderDiscoveryNetwork(torch.nn.Module):
    def __init__(self, node_features, edge_features, classes, node_filters, edge_filters):
        super(PathfinderDiscoveryNetwork, self).__init__()
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
        
        
class Trainer(object):
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def train_model(self, model, dataset):
        print("Training started.\n")
        
        
        edges = dataset["edges"]
        node_features = dataset["node_features"]
        edge_features = dataset["edge_features"]
        target = dataset["target"]
        
        train_index = dataset["train_index"]
        test_index = dataset["test_index"]
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        model.train()
        for epoch in tqdm(range(self.epochs)):
            optimizer.zero_grad()
            prediction = model(node_features,
                               edges,
                               edge_features)
                               
            loss = F.nll_loss(prediction[train_index], target[train_index])
            loss.backward()
            optimizer.step()
        model.eval()
        _, prediction = model(node_features, edges, edge_features).max(dim=1)
        correct = int(prediction[test_index].eq(target[test_index]).sum().item())
        acc = correct / int(test_index.shape[0])
        print('\nAccuracy: {:.4f}'.format(acc))
