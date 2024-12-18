import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, average_precision_score
from tqdm import tqdm
import logging
import random
import mygene
import gseapy as gp
from scipy.stats import ttest_ind

################################################################################
# This code integrates LRP, IG, and GSEA methods into a single workflow.
# The interpretation method can be chosen from "LRP", "IG", or "GSEA" via config.
#
# Methods:
# - LRP: Layer-wise Relevance Propagation (model-based interpretation)
# - IG: Integrated Gradients (model-based interpretation)
# - GSEA: Gene Set Enrichment Analysis (data-driven enrichment analysis)
#
# Note: GSEA is largely independent of the model's predictions. It relies on
# statistical differences in gene expression data between two groups.
################################################################################

# Configuration function
def get_config():
    config = {
        "data_paths": {
            "data_dir": '/home/bml_jk/py/lrpGP/data/alzheimer'
        },
        "file_paths": {
            "gene_expression": 'alzh_GE_data.csv',
            "gene_pathway_mapping": 'alzh_gene_pathway_mapping.csv',
            "gene_pathway_matrix": 'gene_pathway_matrix.txt',
            "input_to_h1_masking": 'masking_input_to_h1.txt',
            "h1_to_pathway_masking": 'masking_h1_to_pathway.txt',
            "model_save": '/home/bml_jk/py/lrpGP/savedModel/ALZH/alzh_valid.pth',
            "result_save": '/home/bml_jk/py/lrpGP/result/alzh/valid_alzh_pathway_info.csv',
            "gsea_result_save": '/home/bml_jk/py/lrpGP/result/revision/gsea_results_alzh.csv',
            "gsea_outdir": '/home/bml_jk/py/lrpGP/result/revision',
            "gsea_library": '/home/bml_jk/py/lrpGP/data/gsea/ReactomePathways.gmt'
        },
        "params": {
            "interpretation_method": "LRP",  # Options: "LRP", "IG", "GSEA"
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

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Logger class
class Logger:
    def __init__(self, process_name=""):
        self.logger = self.initialize_logging(process_name)

    @staticmethod
    def initialize_logging(process_name=""):
        logging.basicConfig(level=logging.INFO, format=f'%(asctime)s - {process_name} - %(levelname)s - %(message)s')
        return logging.getLogger(process_name)

    def info(self, message):
        self.logger.info(message)

# Initialize main logger
main_logger = Logger("Main Process")
set_random_seeds(42)

# DataProcessor class
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
        gene_list = df_filtered_gene_expression.columns.tolist()[:-1]  # Exclude label column
        gene_count = len(gene_list)

        # Count the number of genes in each pathway
        pathway_gene_counts = df_filtered_gene_pathway.groupby('pathway_name')['gene'].nunique().reset_index()
        pathway_gene_counts.columns = ['pathway_name', 'num_of_genes']
        pathway_gene_counts = pathway_gene_counts.sort_values(by='num_of_genes', ascending=False)

        # Filter pathways based on gene thresholds
        sorted_pathway_gene_counts = pathway_gene_counts[
            (pathway_gene_counts['num_of_genes'] >= min_genes_threshold) &
            (pathway_gene_counts['num_of_genes'] <= max_genes_threshold)
        ].sort_values(by='num_of_genes', ascending=False)
        sorted_pathways = sorted_pathway_gene_counts['pathway_name'].tolist()

        # Determine indices for top, middle, and bottom percentages
        top_30_percent_index = int(len(sorted_pathway_gene_counts) * 0.3)
        bottom_40_percent_index = int(len(sorted_pathway_gene_counts) * 0.7)

        hidden_one_list = []
        def create_hidden_nodes(counts, index_range, max_nodes):
            for pathway in counts['pathway_name'].iloc[index_range].tolist():
                genes_in_pathway = df_filtered_gene_pathway[df_filtered_gene_pathway['pathway_name'] == pathway]['gene'].tolist()
                for i in range(1, min(max_nodes + 1, len(genes_in_pathway) + 1)):
                    hidden_one_list.append(f"{pathway}_{i}")

        # Create hidden nodes for top 30% pathways
        create_hidden_nodes(sorted_pathway_gene_counts, slice(0, top_30_percent_index), top_30_nodes)
        # Create hidden nodes for middle 30% pathways
        create_hidden_nodes(sorted_pathway_gene_counts, slice(top_30_percent_index, bottom_40_percent_index), middle_30_nodes)
        # Create hidden nodes for bottom 40% pathways
        create_hidden_nodes(sorted_pathway_gene_counts, slice(bottom_40_percent_index, None), bottom_40_nodes)

        return gene_list, gene_count, sorted_pathways, hidden_one_list

# MatrixHandler class
class MatrixHandler:
    @staticmethod
    def load_or_create_matrices(gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config):
        data_dir = config["data_paths"]["data_dir"]
        gene_pathway_matrix_path = os.path.join(data_dir, config["file_paths"]["gene_pathway_matrix"])
        input_to_h1_masking_path = os.path.join(data_dir, config["file_paths"]["input_to_h1_masking"])
        h1_to_pathway_masking_path = os.path.join(data_dir, config["file_paths"]["h1_to_pathway_masking"])

        # Load or create gene-pathway matrix
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

        input_to_h1_masking = create_or_load_masking_matrix(
            gene_list, hidden_one_list, [h.split('_')[0] for h in hidden_one_list], input_to_h1_masking_path
        )

        h1_to_pathway_masking = create_or_load_masking_matrix(
            hidden_one_list, sorted_pathways, sorted_pathways, h1_to_pathway_masking_path
        )

        return gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking

# Custom dataset
class loadEXPdataset(Dataset):
    def __init__(self, features, labels, device=device):
        self.X_tensor = torch.Tensor(features).to(device)
        self.y_tensor = torch.Tensor(labels).long().to(device)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, index):
        return {'features': self.X_tensor[index], 'labels': self.y_tensor[index]}

# Neural network model
class EXPNet(nn.Module):
    def __init__(self, input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, 
                 input_to_h1_masking, h1_to_pathway_masking, dropout_rate1=0.8, dropout_rate2=0.7):
        super(EXPNet, self).__init__()
        self.hidden_one = nn.Linear(input_dim, hidden_one_dim, bias=True)
        self.pathway_layer = nn.Linear(hidden_one_dim, pathway_dim, bias=True)
        self.hidden_two = nn.Linear(pathway_dim, hidden_two_dim, bias=True)
        self.output_layer = nn.Linear(hidden_two_dim, output_dim, bias=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)

        input_to_h1_masking_tensor = torch.Tensor(input_to_h1_masking.T)
        h1_to_pathway_masking_tensor = torch.Tensor(h1_to_pathway_masking.T)

        self.input_to_h1_masking = input_to_h1_masking_tensor
        self.h1_to_pathway_masking = h1_to_pathway_masking_tensor
        self.device = device

        with torch.no_grad():
            self.hidden_one.weight *= self.input_to_h1_masking
            self.pathway_layer.weight *= self.h1_to_pathway_masking

    def forward(self, x):
        x = self.leaky_relu(self.hidden_one(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.pathway_layer(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.hidden_two(x))
        x = self.output_layer(x)
        return x

# LRPNet for LRP calculation
class LRPNet(EXPNet):
    def forward(self, x):
        x_hidden_one = self.leaky_relu(self.hidden_one(x))
        x_hidden_one_drop = self.dropout1(x_hidden_one)
        x_pathway_layer = self.leaky_relu(self.pathway_layer(x_hidden_one_drop))
        x_pathway_layer_drop = self.dropout2(x_pathway_layer)
        x_hidden_two = self.leaky_relu(self.hidden_two(x_pathway_layer_drop))
        x_output = self.output_layer(x_hidden_two)
        return x_output, x_hidden_one, x_pathway_layer, x_hidden_two

# Custom Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.4, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Model trainer with k-fold
class ModelTrainer:
    def __init__(self, model_class, dataset, model_params, save_model_path, k_folds=5, epochs=200, 
                 batch_size=64, learning_rate=0.001, weight_decay=3e-4, patience=44, 
                 device=device, use_amp=True, save_model=True):
        self.model_class = model_class
        self.dataset = dataset
        self.model_params = model_params
        self.save_model_path = save_model_path
        self.k_folds = k_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.use_amp = use_amp
        self.save_model = save_model 

    def train(self, logger, verbose=True):
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
        fold_results = []
        fold_model_state_dicts = []

        classes = np.unique(self.dataset.y_tensor.cpu().numpy())

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.dataset.X_tensor.cpu().numpy(), self.dataset.y_tensor.cpu().numpy())):
            if verbose:
                logger.info(f'FOLD {fold+1} -------------------------------------')
            
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

            train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=0, pin_memory=False)
            val_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=0, pin_memory=False)

            model = self.model_class(**self.model_params).to(self.device)
            criterion = FocalLoss(gamma=2, alpha=0.9)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scaler = torch.amp.GradScaler(enabled=self.use_amp)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            best_accuracy = 0
            best_roc_auc = 0
            best_precision = 0
            best_f1 = 0
            best_avg_precision = 0
            best_model_state_dict = None

            for epoch in tqdm(range(self.epochs), desc=f"Training Epochs for Fold {fold+1}", disable=not verbose):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                        outputs = model(batch['features'])
                        loss = criterion(outputs, batch['labels'])
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0.0
                val_labels = []
                val_preds = []
                val_probs = []
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                            outputs = model(batch['features'])
                            loss = criterion(outputs, batch['labels'])
                        val_loss += loss.item()
                        probs = torch.softmax(outputs, dim=1)
                        _, preds = torch.max(probs, 1)
                        val_labels.extend(batch['labels'].cpu().numpy())
                        val_preds.extend(preds.cpu().numpy())
                        val_probs.extend(probs.cpu().numpy())

                val_loss /= len(val_loader)
                val_labels = np.array(val_labels)
                val_probs = np.array(val_probs)

                accuracy = accuracy_score(val_labels, val_preds)
                roc_auc = roc_auc_score(val_labels, val_probs[:, 1])
                precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
                f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
                avg_precision = average_precision_score(val_labels, val_probs[:, 1])

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                if precision > best_precision:
                    best_precision = precision
                if f1 > best_f1:
                    best_f1 = f1
                if avg_precision > best_avg_precision:
                    best_avg_precision = avg_precision

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state_dict = model.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break

            fold_results.append((best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision))
            fold_model_state_dicts.append(best_model_state_dict)

            logger.info(f'Fold {fold+1} Results: Accuracy: {best_accuracy:.4f}, ROC AUC: {best_roc_auc:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}, Avg Precision: {best_avg_precision:.4f}')

        # Compute average metrics
        avg_accuracy = np.mean([result[0] for result in fold_results])
        avg_roc_auc = np.mean([result[1] for result in fold_results])
        avg_precision = np.mean([result[2] for result in fold_results])
        avg_f1 = np.mean([result[3] for result in fold_results])
        avg_avg_precision = np.mean([result[4] for result in fold_results])

        best_accuracy = np.max([result[0] for result in fold_results])
        best_roc_auc = np.max([result[1] for result in fold_results])
        best_precision = np.max([result[2] for result in fold_results])
        best_f1 = np.max([result[3] for result in fold_results])
        best_avg_precision = np.max([result[4] for result in fold_results])

        # Select the median model based on ROC AUC
        roc_auc_scores = [result[1] for result in fold_results]
        sorted_indices = np.argsort(roc_auc_scores)
        median_index = sorted_indices[len(sorted_indices) // 2]
        best_model_state_dict = fold_model_state_dicts[median_index]

        if self.save_model and best_model_state_dict is not None:
            torch.save(best_model_state_dict, self.save_model_path)
            logger.info(f"Saved median model from fold {median_index + 1} to {self.save_model_path}")

        return best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision

# LRPHandler for LRP calculations
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
        relevance_hidden_one = LRPHandler.lrp_linear(x_hidden_one, x_pathway_layer, model.pathway_layer.weight, model.pathway_layer.bias, relevance_pathway_layer)
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

# IGHandler for Integrated Gradients
class IGHandler:
    @staticmethod
    def compute_integrated_gradients(model, x, target_class, baseline=None, steps=50, device=device):
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

        for iteration in tqdm(range(num_iterations), desc="Computing Integrated Gradients", leave=False):
            iteration_ig_scores = np.zeros(pathway_dim)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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

# ResultPrinter class
class ResultPrinter:
    @staticmethod
    def print_training_results(logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision):
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
    def save_scores_to_csv(pathway_list, scores, gene_pathway_df, save_path, score_name="LRP_Score"):
        data = []
        for idx, score in enumerate(scores):
            pathway_name = pathway_list[idx]
            num_genes = len(gene_pathway_df[gene_pathway_df['pathway_name'] == pathway_name])
            data.append((pathway_name, score, num_genes))
        df_scores = pd.DataFrame(data, columns=['Pathway_Name', score_name, 'Num_Genes'])
        df_scores.to_csv(save_path, index=False)
        logging.info(f"{score_name} results saved to {save_path}")

# GSEAHandler for GSEA analysis
class GSEAHandler:
    @staticmethod
    def run_gsea_analysis(df_filtered_gene_expression, output_path, gsea_gmt, gsea_params):
        # Preprocessing for GSEA
        # Extract genes and groups
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

# Main execution
if __name__ == "__main__":
    config = get_config()
    main_logger.info("Loading data...")
    df_filtered_gene_expression, df_filtered_gene_pathway = DataProcessor.load_data(config)
    main_logger.info("Data loaded successfully.")

    if config["params"]["interpretation_method"] in ["LRP", "IG"]:
        main_logger.info("Processing data for model-based interpretation...")
        gene_list, gene_count, sorted_pathways, hidden_one_list = DataProcessor.process_data(
            df_filtered_gene_expression, df_filtered_gene_pathway, 
            config["params"]["min_genes_threshold"], config["params"]["max_genes_threshold"],
            config["params"]["top_30_nodes"], config["params"]["middle_30_nodes"], config["params"]["bottom_40_nodes"]
        )
        main_logger.info("Data processed successfully.")

        main_logger.info("Loading or creating matrices...")
        gene_pathway_matrix, input_to_h1_masking, h1_to_pathway_masking = MatrixHandler.load_or_create_matrices(
            gene_list, gene_count, sorted_pathways, df_filtered_gene_pathway, hidden_one_list, config
        )
        main_logger.info("Matrices loaded/created successfully.")

        main_logger.info("Loading features and labels...")
        features = df_filtered_gene_expression[gene_list].values
        labels = df_filtered_gene_expression['label'].values

        main_logger.info("Creating dataset...")
        dataset = loadEXPdataset(features, labels)

        main_logger.info("Defining model parameters...")
        input_dim = gene_count
        hidden_one_dim = len(hidden_one_list)
        pathway_dim = len(sorted_pathways)
        hidden_two_dim = 128
        output_dim = len(np.unique(labels))

        model_params = {
            'input_dim': input_dim,
            'hidden_one_dim': hidden_one_dim,
            'pathway_dim': pathway_dim,
            'hidden_two_dim': hidden_two_dim,
            'output_dim': output_dim,
            'input_to_h1_masking': input_to_h1_masking,
            'h1_to_pathway_masking': h1_to_pathway_masking
        }

        save_model_path = config["file_paths"]["model_save"]

        # Use EXPNet for training
        main_logger.info("Initializing ModelTrainer for training...")
        model_trainer = ModelTrainer(EXPNet, dataset, model_params, save_model_path, k_folds=config["training"]["k_folds"], 
                                        epochs=200, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4,
                                        patience=config["training"]["patience"], device=device, use_amp=True, save_model=True)

        main_logger.info("Starting training...")
        best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision = model_trainer.train(main_logger)

        # Load trained model
        if config["params"]["interpretation_method"] == "LRP":
            main_logger.info("Running LRP interpretation...")
            lrp_model = LRPNet(**model_params).to(device)
            lrp_model.load_state_dict(best_model_state_dict)
            obtained_scores = LRPHandler.compute_lrp_for_dataset_multiple_times(
                lrp_model, dataset, target_class=config["params"]["target_class"], num_iterations=config["params"]["num_lrp_iterations"]
            )
            top_pathways, top_indices = LRPHandler.get_top_lrp_scores(obtained_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
            ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision)
            ResultPrinter.print_top_scores(main_logger, top_pathways, method_name="LRP")
            csv_save_path = config["file_paths"]["result_save"].replace(".csv", "_LRP.csv")
            ResultPrinter.save_scores_to_csv(sorted_pathways, obtained_scores, df_filtered_gene_pathway, csv_save_path, score_name="LRP_Score")

        elif config["params"]["interpretation_method"] == "IG":
            main_logger.info("Running IG interpretation...")
            ig_model = EXPNet(**model_params).to(device)
            ig_model.load_state_dict(best_model_state_dict)
            obtained_scores = IGHandler.compute_integrated_gradients_for_dataset_multiple_times(
                ig_model, dataset, gene_pathway_matrix, target_class=config["params"]["target_class"], num_iterations=config["params"]["num_ig_iterations"]
            )
            top_pathways, top_indices = IGHandler.get_top_ig_scores(obtained_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
            ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision)
            ResultPrinter.print_top_scores(main_logger, top_pathways, method_name="IG")
            csv_save_path = config["file_paths"]["result_save"].replace(".csv", "_IG.csv")
            ResultPrinter.save_scores_to_csv(sorted_pathways, obtained_scores, df_filtered_gene_pathway, csv_save_path, score_name="IG_Score")

    elif config["params"]["interpretation_method"] == "GSEA":
        main_logger.info("Running GSEA analysis...")
        # GSEA does not require model training, it's a separate analysis
        gsea_results = GSEAHandler.run_gsea_analysis(
            df_filtered_gene_expression,
            config["file_paths"]["gsea_outdir"],
            config["file_paths"]["gsea_library"],
            config["gsea_params"]
        )
        res_df = gsea_results.res2d
        GSEAHandler.print_gsea_results(res_df)
        # Save results
        res_df.to_csv(config["file_paths"]["gsea_result_save"], index=False)
        main_logger.info(f"GSEA results saved to {config['file_paths']['gsea_result_save']}")

    else:
        main_logger.info("Invalid interpretation method specified.")