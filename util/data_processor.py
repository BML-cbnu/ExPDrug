import os
import pandas as pd

class DataProcessor:
    @staticmethod
    def load_data(config):
        data_dir = config["data_paths"]["data_dir"]
        gene_expression_path = os.path.join(data_dir, config["file_paths"]["gene_expression"])
        gene_pathway_mapping_path = os.path.join(data_dir, config["file_paths"]["gene_pathway_mapping"])
        
        df_filtered_gene_expression = pd.read_csv(gene_expression_path)
        df_filtered_gene_pathway = pd.read_csv(gene_pathway_mapping_path)
        return df_filtered_gene_expression, df_filtered_gene_pathway

    @staticmethod
    def process_data(df_filtered_gene_expression, df_filtered_gene_pathway, min_genes_threshold, max_genes_threshold, top_30_nodes, middle_30_nodes, bottom_40_nodes):
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
        def create_hidden_nodes(counts, index_range, max_nodes):
            for pathway in counts['pathway_name'].iloc[index_range].tolist():
                genes_in_pathway = df_filtered_gene_pathway[df_filtered_gene_pathway['pathway_name'] == pathway]['gene'].tolist()
                for i in range(1, min(max_nodes + 1, len(genes_in_pathway) + 1)):
                    hidden_one_list.append(f"{pathway}_{i}")

        create_hidden_nodes(sorted_pathway_gene_counts, slice(0, top_30_percent_index), top_30_nodes)
        create_hidden_nodes(sorted_pathway_gene_counts, slice(top_30_percent_index, bottom_40_percent_index), middle_30_nodes)
        create_hidden_nodes(sorted_pathway_gene_counts, slice(bottom_40_percent_index, None), bottom_40_nodes)

        return gene_list, gene_count, sorted_pathways, hidden_one_list