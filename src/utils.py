import numpy as np
from texttable import Texttable

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
    
    def _read_edges(self):
        self.edges = read_array(self._edges_path)
        self.edge_count = self.edges_shape[0]
        self.edge_feature_count = self.edges.shape[1]
        
    def _read_node_features(self):
        self.node_features = read_array(self._node_features_path)
        self.node_count = self.node_features.shape[0]
        self.node_feature_count = self.node_features.shape[1]
        
    def _read_edge_features(self):
        self.edge_features = read_array(self._edge_features_path)
        self.edge_feature_count = self.edge_features.shape[1]
        
        
    def _read_target(self):
        self.target = read_array(self._target_path)
        self.number_of_classes = np.max(target) + 1
        
        
    def get_dataset(self):
        self._read_edges()
        self._read_node_features()
        self._read_edge_features()
        self._read_target()
