import logging
import torch
import random
import numpy as np
import pandas as pd

def get_config():
    config = {
        "data_paths": {
            "data_dir": 'data'
        },
        "file_paths": {
            "gene_expression": 'gene_expression.csv',
            "gene_pathway_mapping": 'gene_pathway_mapping.csv',
            "gene_pathway_matrix": 'gene_pathway_matrix.txt',
            "input_to_h1_masking": 'masking_input_to_h1.txt',
            "h1_to_pathway_masking": 'masking_h1_to_pathway.txt',
            "model_save": 'result/model_best.pth',
            "result_save": 'result/valid_pathway_info.csv',
            "gsea_result_save": 'result/gsea_results.csv',
            "gsea_outdir": 'result/gsea_output',
            "gsea_library": 'data/ReactomePathways.gmt'
        },
        "params": {
            "interpretation_method": "LRP",  # "LRP", "IG", or "GSEA"
            "min_genes_threshold": 1,
            "max_genes_threshold": 1300,
            "target_class": 1,
            "num_lrp_iterations": 100,
            "num_ig_iterations": 100,
            "top_30_nodes": 4,
            "middle_30_nodes": 2,
            "bottom_40_nodes": 1
        },
        "training": {
            "k_folds": 5,
            "batch_size": 64,
            "patience": 100
        },
        "loss": {
            "type": 'FocalLoss',
            "gamma": 2,
            "alpha": 0.9
        },
        "gsea_params": {
            "permutation_num": 1000,
            "seed": 42
        }
    }
    return config

def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger:
    def __init__(self, process_name=""):
        self.logger = self.initialize_logging(process_name)

    @staticmethod
    def initialize_logging(process_name=""):
        logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - {process_name} - %(levelname)s - %(message)s')
        return logging.getLogger(process_name)

    def info(self, message):
        self.logger.info(message)

class ResultPrinter:
    @staticmethod
    def print_training_results(logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision,
                               best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision):
        logger.info('--------------------------------')
        logger.info(f'Average Accuracy: {avg_accuracy}')
        logger.info(f'Average ROC AUC: {avg_roc_auc}')
        logger.info(f'Average Precision: {avg_precision}')
        logger.info(f'Average F1 Score: {avg_f1}')
        logger.info(f'Average AU-PRC: {avg_avg_precision}')
        logger.info('--------------------------------')
        logger.info(f'Best Accuracy: {best_accuracy}')
        logger.info(f'Best ROC AUC: {best_roc_auc}')
        logger.info(f'Best Precision: {best_precision}')
        logger.info(f'Best F1 Score: {best_f1}')
        logger.info(f'Best AU-PRC: {best_avg_precision}')

    @staticmethod
    def print_top_scores(logger, top_pathways, method_name="LRP"):
        logger.info(f"Top {len(top_pathways)} {method_name} scores and corresponding pathways:")
        logger.info("----------------------------------------")
        for pathway_name, score, gene_count in top_pathways:
            logger.info(f"Pathway: {pathway_name}, Score: {score}, Number of Genes: {gene_count}")

    @staticmethod
    def save_scores_to_csv(pathway_list, scores, gene_pathway_df, save_path, score_name="Score"):
        data = []
        for idx, score in enumerate(scores):
            pathway_name = pathway_list[idx]
            num_genes = len(gene_pathway_df[gene_pathway_df['pathway_name'] == pathway_name])
            data.append((pathway_name, score, num_genes))
        df_scores = pd.DataFrame(data, columns=['Pathway_Name', score_name, 'Num_Genes'])
        df_scores.to_csv(save_path, index=False)
        logging.info(f"{score_name} results saved to {save_path}")