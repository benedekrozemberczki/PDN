import torch
from utils import tab_printer, PathfinderDatasetReader
from param_parser import parameter_parser


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
    reader.create_split()
    dataset = reader.get_dataset()
    
    
if __name__ == "__main__":
    main()
