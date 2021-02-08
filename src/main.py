import torch
from param_parser import parameter_parser
from pdn import PathfinderDiscoveryNetwork
from utils import tab_printer, PathfinderDatasetReader


def main():
    """
    Parsing command line parameters, reading data, fitting a PDN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    dataset = PathfinderDataset(args.edges_path,
                                args.node_features_path,
                                args.edge_features_path,
                                args.target_path)
    reader.read_dataset()
    reader.create_split(args.test_size, args.seed)
    dataset = reader.get_dataset()
    model = PathfinderDiscoveryNetwork(data["node_feature_count"], data["edge_feature_count"], data["classes"]
                                       args.node_filters, args.edge_filters)
    
    
if __name__ == "__main__":
    main()
