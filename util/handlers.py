import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import gseapy as gp
import mygene
from scipy.stats import ttest_ind
import logging

class LRPHandler:
    @staticmethod
    def lrp_linear(h_in, h_out, weight, bias, relevance_out, epsilon=1e-6):
        z = h_in @ weight.t() + bias
        s = relevance_out / (z + epsilon)
        c = s @ weight
        relevance_in = h_in * c
        return relevance_in

    @staticmethod
    def compute_lrp(model, x, target_class, epsilon=1e-6):
        model.eval()
        output, x_hidden_one, x_pathway_layer, x_hidden_two = model(x)
        relevance_output = torch.zeros_like(output).to(model.device)
        relevance_output[:, target_class] = output[:, target_class].clamp(min=epsilon)

        relevance_hidden_two = LRPHandler.lrp_linear(x_hidden_two, output, model.output_layer.weight, model.output_layer.bias, relevance_output)
        relevance_pathway_layer = LRPHandler.lrp_linear(x_pathway_layer, x_hidden_two, model.hidden_two.weight, model.hidden_two.bias, relevance_hidden_two)
        # relevance_hidden_one is computed but not returned in this step as we only need pathway layer relevance
        return relevance_pathway_layer

    @staticmethod
    def compute_lrp_for_dataset_multiple_times(model, dataset, batch_size=32, num_iterations=100, target_class=1):
        pathway_dim = model.pathway_layer.out_features
        all_lrp_scores = np.zeros((num_iterations, pathway_dim))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for iteration in tqdm(range(num_iterations), desc="Computing LRP scores", leave=False):
            iteration_lrp_scores = np.zeros(pathway_dim)
            for batch in data_loader:
                sample_inputs = batch['features']
                lrp_scores = LRPHandler.compute_lrp(model, sample_inputs, target_class)
                iteration_lrp_scores += lrp_scores.cpu().detach().numpy().sum(axis=0)
            all_lrp_scores[iteration] = iteration_lrp_scores / len(dataset)
        
        return np.mean(all_lrp_scores, axis=0)

    @staticmethod
    def get_top_lrp_scores(lrp_scores, pathway_list, gene_pathway_df, top_n=10):
        lrp_scores = lrp_scores.flatten()
        top_indices = np.argsort(lrp_scores)[-top_n:][::-1]
        top_pathways = []
        for idx in top_indices:
            pathway_name = pathway_list[idx]
            score = lrp_scores[idx]
            num_genes = len(gene_pathway_df[gene_pathway_df['pathway_name'] == pathway_name])
            top_pathways.append((pathway_name, score, num_genes))
        return top_pathways, top_indices


class IGHandler:
    @staticmethod
    def compute_integrated_gradients(model, x, target_class, baseline=None, steps=50, device=torch.device("cpu")):
        if baseline is None:
            baseline = torch.zeros_like(x).to(device)
        else:
            baseline = baseline.to(device)
        scaled_inputs = [baseline + (float(i)/steps)*(x - baseline) for i in range(steps+1)]
        gradients = []
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad_(True)
            output = model(scaled_input)
            output = output[:, target_class]
            model.zero_grad()
            output.backward(torch.ones_like(output))
            grad = scaled_input.grad.detach().clone()
            gradients.append(grad)
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_gradients = (x - baseline) * avg_gradients
        return integrated_gradients

    @staticmethod
    def compute_integrated_gradients_for_dataset_multiple_times(model, dataset, gene_pathway_matrix, batch_size=32, num_iterations=100, target_class=1, steps=50):
        pathway_dim = gene_pathway_matrix.shape[1]
        all_ig_scores = np.zeros((num_iterations, pathway_dim))

        gene_pathway_matrix = torch.tensor(gene_pathway_matrix, dtype=torch.float32).to(model.device)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for iteration in tqdm(range(num_iterations), desc="Computing Integrated Gradients", leave=False):
            iteration_ig_scores = np.zeros(pathway_dim)
            for batch in data_loader:
                sample_inputs = batch['features']
                ig_attributions = IGHandler.compute_integrated_gradients(model, sample_inputs, target_class, steps=steps, device=model.device)
                gene_attributions = ig_attributions.detach().cpu().numpy()
                pathway_attributions = gene_attributions @ gene_pathway_matrix.cpu().numpy()
                iteration_ig_scores += pathway_attributions.sum(axis=0)
            all_ig_scores[iteration] = iteration_ig_scores / len(dataset)
        return np.mean(all_ig_scores, axis=0)

    @staticmethod
    def get_top_ig_scores(ig_scores, pathway_list, gene_pathway_df, top_n=10):
        ig_scores = ig_scores.flatten()
        top_indices = np.argsort(ig_scores)[-top_n:][::-1]
        top_pathways = []
        for idx in top_indices:
            pathway_name = pathway_list[idx]
            score = ig_scores[idx]
            num_genes = len(gene_pathway_df[gene_pathway_df['pathway_name'] == pathway_name])
            top_pathways.append((pathway_name, score, num_genes))
        return top_pathways, top_indices


class GSEAHandler:
    @staticmethod
    def run_gsea_analysis(df_filtered_gene_expression, output_path, gsea_gmt, gsea_params):
        # Preprocessing for GSEA
        gene_list = df_filtered_gene_expression.columns.tolist()
        gene_list.remove('label')
        group0 = df_filtered_gene_expression[df_filtered_gene_expression['label'] == 0]
        group1 = df_filtered_gene_expression[df_filtered_gene_expression['label'] == 1]

        # Handle negative values
        min_value = df_filtered_gene_expression[gene_list].min().min()
        if min_value <= 0:
            offset = abs(min_value) + 1
            df_filtered_gene_expression[gene_list] += offset

        # Re-split groups after offset
        group0 = df_filtered_gene_expression[df_filtered_gene_expression['label'] == 0]
        group1 = df_filtered_gene_expression[df_filtered_gene_expression['label'] == 1]

        # Compute stats
        mean_group0 = group0[gene_list].mean()
        mean_group1 = group1[gene_list].mean()
        fold_changes = np.log2((mean_group1 + 1e-8) / (mean_group0 + 1e-8))
        t_stats, p_values = ttest_ind(group1[gene_list], group0[gene_list], equal_var=False)

        # Remove NaN
        valid_indices = np.isfinite(t_stats)
        t_stats = t_stats[valid_indices]
        p_values = p_values[valid_indices]
        valid_genes = np.array(gene_list)[valid_indices]
        fold_changes = fold_changes[valid_genes]

        deg = pd.DataFrame({
            'gene': valid_genes,
            'log2FC': fold_changes.values,
            't_stat': t_stats,
            'p_value': p_values
        })

        # Convert gene names to uppercase
        deg['gene'] = deg['gene'].str.upper()

        # Map gene names using mygene
        mg = mygene.MyGeneInfo()
        query_results = mg.querymany(deg['gene'].tolist(), scopes='symbol', fields='symbol', species='human')
        gene_symbol_mapping = {entry['query']: entry.get('symbol', entry['query']) for entry in query_results}
        deg['gene'] = deg['gene'].map(gene_symbol_mapping)

        deg_sorted = deg.sort_values(by='t_stat', ascending=False)
        gene_ranking = deg_sorted.set_index('gene')['t_stat']

        # Run GSEA prerank
        gsea_results = gp.prerank(
            rnk=gene_ranking,
            gene_sets=gsea_gmt,
            outdir=output_path,
            format='png',
            permutation_num=gsea_params['permutation_num'],
            seed=gsea_params['seed']
        )
        return gsea_results

    @staticmethod
    def print_gsea_results(res_df):
        if 'NES' in res_df.columns:
            top_pathways = res_df.sort_values(by='NES', ascending=False).head(20)
            logging.info("Top 20 enriched pathways:")
            logging.info(top_pathways[['Term', 'NES', 'NOM p-val', 'FDR q-val']])
            total_pathways = len(res_df)
            logging.info(f"Total number of pathways considered: {total_pathways}")
        else:
            logging.info("No 'NES' column found in res_df. GSEA may not have returned any results.")