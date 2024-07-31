import os
import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def load_data(config):
        """
        Load gene expression and gene-pathway mapping data from CSV files.
        """
        data_dir = config["data_paths"]["data_dir"]
        gene_expression_path = os.path.join(data_dir, config["file_paths"]["gene_expression"])
        gene_pathway_mapping_path = os.path.join(data_dir, config["file_paths"]["gene_pathway_mapping"])
        
        df_filtered_gene_expression = pd.read_csv(gene_expression_path)
        df_filtered_gene_pathway = pd.read_csv(gene_pathway_mapping_path)
        return df_filtered_gene_expression, df_filtered_gene_pathway

    @staticmethod
    def process_data(df_filtered_gene_expression, df_filtered_gene_pathway, min_genes_threshold, max_genes_threshold, top_30_nodes, middle_30_nodes, bottom_40_nodes):
        """
        Process data to filter pathways and create hidden layer nodes.
        """
        gene_list = df_filtered_gene_expression.columns.tolist()[:-1]
        gene_count = len(gene_list)

        pathway_gene_counts = df_filtered_gene_pathway.groupby('pathway_name')['gene'].nunique().reset_index()
        pathway_gene_counts.columns = ['pathway_name', 'num_of_genes']
        pathway_gene_counts = pathway_gene_counts.sort_values(by='num_of_genes', ascending=False)

        sorted_pathway_gene_counts = pathway_gene_counts[
            (pathway_gene_counts['num_of_genes'] >= min_genes_threshold) &
            (pathway_gene_counts['num_of_genes'] <= max_genes_threshold)
        ].sort_values(by='num_of_genes', ascending=False)
        sorted_pathways = sorted_pathway_gene_counts['pathway_name'].tolist()

        top_30_percent_index = int(len(sorted_pathway_gene_counts) * 0.3)
        bottom_40_percent_index = int(len(sorted_pathway_gene_counts) * 0.7)

        hidden_one_list = []

        def create_hidden_nodes(counts, index_range, node_prefix, max_nodes):
            for pathway in counts['pathway_name'].iloc[index_range].tolist():
                genes_in_pathway = df_filtered_gene_pathway[df_filtered_gene_pathway['pathway_name'] == pathway]['gene'].tolist()
                for i in range(1, min(max_nodes + 1, len(genes_in_pathway) + 1)):
                    hidden_one_list.append(f"{pathway}_{i}")

        create_hidden_nodes(sorted_pathway_gene_counts, slice(0, top_30_percent_index), 'top_30', top_30_nodes)
        create_hidden_nodes(sorted_pathway_gene_counts, slice(top_30_percent_index, bottom_40_percent_index), 'middle_30', middle_30_nodes)
        create_hidden_nodes(sorted_pathway_gene_counts, slice(bottom_40_percent_index, None), 'bottom_40', bottom_40_nodes)

        return gene_list, gene_count, sorted_pathways, hidden_one_list

class MatrixHandler:
    @staticmethod
    def load_or_create_matrices(gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config):
        """
        Load or create gene-pathway and masking matrices.
        """
        gene_pathway_matrix_path = os.path.join(config["data_paths"]["data_dir"], config["file_paths"]["gene_pathway_matrix"])
        input_to_h1_masking_path = os.path.join(config["data_paths"]["data_dir"], config["file_paths"]["input_to_h1_masking"])
        h1_to_pathway_masking_path = os.path.join(config["data_paths"]["data_dir"], config["file_paths"]["h1_to_pathway_masking"])

        if os.path.exists(gene_pathway_matrix_path):
            gene_pathway_matrix = np.loadtxt(gene_pathway_matrix_path, dtype=int)
        else:
            gene_pathway_matrix = np.zeros((gene_count, len(sorted_pathways)), dtype=int)
            for i, gene in enumerate(gene_list):
                gene_pathways = df_filtered_gene_pathway[df_filtered_gene_pathway['gene'] == gene]['pathway_name'].tolist()
                for j, pathway in enumerate(sorted_pathways):
                    if pathway in gene_pathways:
                        gene_pathway_matrix[i, j] = 1
            np.savetxt(gene_pathway_matrix_path, gene_pathway_matrix, fmt='%d')

        def create_or_load_masking_matrix(source_list, target_list, target_names, file_path):
            if os.path.exists(file_path):
                return np.loadtxt(file_path, dtype=int)
            else:
                masking_matrix = np.zeros((len(source_list), len(target_list)), dtype=int)
                for i, source in enumerate(source_list):
                    pathway_name = source.split('_')[0]
                    for j, target in enumerate(target_list):
                        if pathway_name == target_names[j]:
                            masking_matrix[i, j] = 1
                np.savetxt(file_path, masking_matrix, fmt='%d')
                return masking_matrix

        input_to_h1_masking = create_or_load_masking_matrix(gene_list, hidden_one_list, [h.split('_')[0] for h in hidden_one_list], input_to_h1_masking_path)
        h1_to_pathway_masking = create_or_load_masking_matrix(hidden_one_list, sorted_pathways, sorted_pathways, h1_to_pathway_masking_path)

        return gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking

    @staticmethod
    def shuffle_matrix(matrix):
        """
        Shuffle the given matrix.
        """
        shuffled_matrix = matrix.copy()
        np.random.shuffle(shuffled_matrix)
        return shuffled_matrix

    @staticmethod
    def create_shuffled_input_to_h1_masking(gene_pathway_matrix, h1_to_pathway_masking):
        """
        Create a shuffled version of the input to hidden layer 1 masking matrix.
        """
        shuffled_gene_pathway_matrix = MatrixHandler.shuffle_matrix(gene_pathway_matrix)
        shuffled_input_to_h1_masking = np.dot(shuffled_gene_pathway_matrix, h1_to_pathway_masking.T)
        return shuffled_input_to_h1_masking

from torch.utils.data import Dataset
import torch

class loadEXPdataset(Dataset):
    """
    PyTorch Dataset class for loading gene expression data.
    """
    def __init__(self, features, labels, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.X_tensor = torch.Tensor(features).to(device)
        self.y_tensor = torch.Tensor(labels).long().to(device)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, index):
        return {'features': self.X_tensor[index], 'labels': self.y_tensor[index]}
