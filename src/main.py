import torch
from utils import tab_printer, PathfinderDataset
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
    dataset.read_dataset()
    #graph = graph_reader(args.edge_path)
    #features = feature_reader(args.features_path)
    #target = target_reader(args.target_path)
    #clustering_machine = ClusteringMachine(args, graph, features, target)
    #clustering_machine.decompose()
    #gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
    #gcn_trainer.train()
    #gcn_trainer.test()

if __name__ == "__main__":
    main()
