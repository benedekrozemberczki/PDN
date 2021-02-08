import torch
import numpy as np
from texttable import Texttable
from sklearn.model_selection import train_test_split

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())
    
    
def read_array(path):
    """
    Function to read the edge, feature and target arrays.
    :param path: Path to the edge list.
    :return data: Data numpy array
    """
    data = np.load(path)
    return data
    
class PathfinderDataset(object):

    def __init__(self, edges_path, node_features_path,
                 edge_features_path, target_path):

        self._edges_path = edges_path
        self._node_features_path = node_features_path
        self._edge_features_path = edge_features_path
        self._target_path = target_path
        
    def read_dataset(self):
        self.edges = read_array(self._edges_path)
        self.node_features = read_array(self._node_features_path)
        self.edge_features = read_array(self._edge_features_path)
        self.target = read_array(self._target_path)
        
    def create_split(self, test_size, seed):
        self.train_i, self.test_i = train_test_split(np.arange(self.node_features.shape), 
                                                     test_size=test_size,
                                                     random_state=seed)
        
        
    def get_dataset(self):
    
        dataset = {}
        
        dataset["edges"] = torch.LongTensor(self._edges)
        dataset["node_features"] = torch.FloatTensor(self._node_features)
        dataset["edge_features"] = torch.FloatTensor(self._edge_features)
        dataset["target"] = torch.LongTensor(self._target)
        
        dataset["node_feature_count"] = self.node_features.shape[1]
        dataset["edge_feature_count"] = self.edge_features.shape[1]
        dataset["classes"] = np.max(self.target) + 1
        return dataset
         
