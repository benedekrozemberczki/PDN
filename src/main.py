import torch
from param_parser import parameter_parser
from pdn import PathfinderDiscoveryNetwork, Trainer
from utils import tab_printer, PathfinderDatasetReader


def main():
    """
    Parsing command line parameters, reading data, fitting a PDN and scoring the model.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)
    
    reader = PathfinderDatasetReader(args.edges_path,
                                     args.node_features_path,
                                     args.edge_features_path,
                                     args.target_path)
    reader.read_dataset()
    reader.create_split(args.test_size, args.seed)
    dataset = reader.get_dataset()
    
    model = PathfinderDiscoveryNetwork(dataset["node_feature_count"],
                                       dataset["edge_feature_count"],
                                       dataset["classes"],
                                       args.node_filters,
                                       args.edge_filters)
    trainer = Trainer(args.epochs, args.learning_rate)
    trainer.train_model(model, dataset)
    
    
if __name__ == "__main__":
    main()
