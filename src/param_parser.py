import argparse

def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on a synthetic dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--edges-path",
                        nargs = "?",
                        default = "./input/edges.npy",
	                help = "Edges array.")

    parser.add_argument("--node-features-path",
                        nargs = "?",
                        default = "./input/node_features.npy",
	                help = "Node features array.")
	                
    parser.add_argument("--edge-features-path",
                        nargs = "?",
                        default = "./input/edge_features.npy",
	                help = "Edge features array.")	              

    parser.add_argument("--target-path",
                        nargs = "?",
                        default = "./input/target.npy",
	                help = "Target classes array.")
	                
    parser.add_argument("--seed",
                        type = int,
                        default = 42,
	                help = "Random seed. Default is 42.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 200,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--gcn-dropout",
                        type = float,
                        default = 0.5,
	                help = "GCN layer dropout parameter. Default is 0.5.")

    parser.add_argument("--pdn-dropout",
                        type = float,
                        default = 0.5,
	                help = "PDN layer dropout parameter. Default is 0.5.")	                     

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--test-ratio",
                        type = float,
                        default = 0.9,
	                help = "Test data ratio. Default is 0.1.")
    
    return parser.parse_args()
