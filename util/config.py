import torch
import numpy as np
import random

def get_config():
    """
    Configuration settings for the deep learning model.
    Update the file paths to your specific directories and files.
    """
    config = {
        "data_paths": {
            "data_dir": '/set/your/data/directory/here/'  # Directory where the data is stored
        },
        "file_paths": {
            "gene_expression": 'set/your/path/to/your_file.csv',  # Path to gene expression data file
            "gene_pathway_mapping": 'set/your/path/to/your_file.csv',  # Path to gene-pathway mapping data file
            "gene_pathway_matrix": 'set/your/path/to/your_file.txt',  # Path to gene-pathway matrix file
            "input_to_h1_masking": 'set/your/path/to/your_file.txt',  # Path to masking matrix from input to hidden layer 1
            "h1_to_pathway_masking": 'set/your/path/to/your_file.txt',  # Path to masking matrix from hidden layer 1 to pathway layer
            "model_save": '/set/your/path/to/your_model.pth',  # Path to save the trained model
            "result_save": '/set/your/path/to/your_results.csv'  # Path to save the results
        },
        "params": {
            "min_genes_threshold": 1,  # Minimum number of genes threshold for filtering pathways
            "max_genes_threshold": 10000,  # Maximum number of genes threshold for filtering pathways
            "target_class": 1,  # Target class for classification
            "num_permutation_iterations": 10,  # Number of iterations for permutation tests
            "num_lrp_iterations": 10,  # Number of iterations for Layer-wise Relevance Propagation (LRP)
            "top_30_nodes": 333,  # Weight for top 30% nodes
            "middle_30_nodes": 222,  # Weight for middle 30% nodes
            "bottom_40_nodes": 111  # Weight for bottom 40% nodes
        },
        "training": {
            "k_folds": 5,  # Number of folds for cross-validation
            "batch_size": 4,  # Batch size for training
            "patience": 100  # Patience for early stopping
        },
        "loss": {
            "type": 'FocalLoss',  # Type of loss function to use
            "gamma": 2,  # Gamma parameter for Focal Loss
            "alpha": 0.9  # Alpha parameter for Focal Loss
        }
    }
    return config

def set_random_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
