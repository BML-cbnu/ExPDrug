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

""" alzh : target_class = 1 : normal_class = 0 """

# Configuration function
def get_config():
    config = {
        "data_paths": {
            "data_dir": '/home/bml_jk/py/lrpGP/data/alzheimer'
        },
        "file_paths": {
            "gene_expression": '/home/bml_jk/py/lrpGP/data/alzheimer/alzh_GE_data.csv',
            "gene_pathway_mapping": '/home/bml_jk/py/lrpGP/data/alzheimer/alzh_gene_pathway_mapping.csv',
            "gene_pathway_matrix": 'gene_pathway_matrix.txt',
            "input_to_h1_masking": 'masking_input_to_h1.txt',
            "h1_to_pathway_masking": 'masking_h1_to_pathway.txt',
            "model_save": '/home/bml_jk/py/lrpGP/savedModel/ALZH/alzh_valid.pth',
            "result_save": '/home/bml_jk/py/lrpGP/result/alzh/valid_alzh_pathway_info.csv'
        },
        "params": {
            "min_genes_threshold": 1,
            "max_genes_threshold": 1300,
            "target_class": 1,
            "num_permutation_iterations": 1000,
            "num_lrp_iterations": 100,
            "top_30_nodes": 4,  # Configurable
            "middle_30_nodes": 2,  # Configurable
            "bottom_40_nodes": 1  # Configurable
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
            # "type": 'CrossEntropyLoss'
        }
    }
    return config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

main_logger = Logger("Main Process")
set_random_seeds(42)

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
        shuffled_matrix = matrix.copy()
        np.random.shuffle(shuffled_matrix)
        return shuffled_matrix

    @staticmethod
    def create_shuffled_input_to_h1_masking(gene_pathway_matrix, h1_to_pathway_masking):
        shuffled_gene_pathway_matrix = MatrixHandler.shuffle_matrix(gene_pathway_matrix)
        shuffled_input_to_h1_masking = np.dot(shuffled_gene_pathway_matrix, h1_to_pathway_masking.T)
        return shuffled_input_to_h1_masking

class loadEXPdataset(Dataset):
    def __init__(self, features, labels, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.X_tensor = torch.Tensor(features).to(device)
        self.y_tensor = torch.Tensor(labels).long().to(device)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, index):
        return {'features': self.X_tensor[index], 'labels': self.y_tensor[index]}

class EXPNet(nn.Module):
    def __init__(self, input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1=0.8, dropout_rate2=0.7): #0.8, 0.7
        super(EXPNet, self).__init__()
        self.hidden_one = nn.Linear(input_dim, hidden_one_dim, bias=True)
        self.pathway_layer = nn.Linear(hidden_one_dim, pathway_dim, bias=True)
        self.hidden_two = nn.Linear(pathway_dim, hidden_two_dim, bias=True)
        self.output_layer = nn.Linear(hidden_two_dim, output_dim, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)

        input_to_h1_masking_tensor = torch.Tensor(input_to_h1_masking.T)
        h1_to_pathway_masking_tensor = torch.Tensor(h1_to_pathway_masking.T)

        with torch.no_grad():
            self.hidden_one.weight *= input_to_h1_masking_tensor
            self.pathway_layer.weight *= h1_to_pathway_masking_tensor

    def forward(self, x):
        x = self.leaky_relu(self.hidden_one(x))
        x = self.dropout1(x)  # Apply first dropout
        x = self.leaky_relu(self.pathway_layer(x))
        x = self.dropout2(x)  # Apply second dropout
        x = self.leaky_relu(self.hidden_two(x))
        x = self.output_layer(x)
        return x

class LRPNet(EXPNet):
    def __init__(self, input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1=0.8, dropout_rate2=0.7):
        super(LRPNet, self).__init__(input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1, dropout_rate2)
    
    def forward(self, x):
        x_hidden_one = self.leaky_relu(self.hidden_one(x))
        x_hidden_one = self.dropout1(x_hidden_one)
        x_pathway_layer = self.leaky_relu(self.pathway_layer(x_hidden_one))
        x_pathway_layer = self.dropout2(x_pathway_layer)
        x_hidden_two = self.leaky_relu(self.hidden_two(x_pathway_layer))
        x_output = self.output_layer(x_hidden_two)
        return x_output, x_hidden_one, x_pathway_layer, x_hidden_two

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

class ModelTrainer:
    def __init__(self, model_class, dataset, model_params, save_model_path, k_folds=10, epochs=1000, 
                 batch_size=4, learning_rate=0.001, weight_decay=3e-4, patience=44, 
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), use_amp=False, save_model=True):
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
            criterion = FocalLoss(gamma=2, alpha=0.9)  # Use Focal Loss
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)  # Apply L2 regularization
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            best_accuracy = 0
            best_roc_auc = 0
            best_precision = 0
            best_f1 = 0
            best_avg_precision = 0
            early_stopping_epoch = 0

            for epoch in tqdm(range(self.epochs), desc=f"Training Epochs for Fold {fold+1}", disable=not verbose):
                model.train()
                train_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = model(batch['features'])
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
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
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            outputs = model(batch['features'])
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
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
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        early_stopping_epoch = epoch + 1
                        break

            fold_results.append((best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision))
            fold_model_state_dicts.append(model.state_dict())

            # Print each fold's results
            logger.info(f'Fold {fold+1} Results: Accuracy: {best_accuracy:.4f}, ROC AUC: {best_roc_auc:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}, Avg Precision: {best_avg_precision:.4f}')

        # After all folds have been processed, compute average metrics
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

        # Save the median model and log which fold was saved
        if self.save_model and best_model_state_dict is not None:
            torch.save(best_model_state_dict, self.save_model_path)
            logger.info(f"Saved median model from fold {median_index + 1} to {self.save_model_path}")

        return best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision

class p_Test:
    def __init__(self, model_class, dataset, model_params, best_model_path, 
                 epochs=1000, batch_size=4, learning_rate=0.001, weight_decay=3e-4,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), use_amp=False):
        self.model_class = model_class
        self.dataset = dataset
        self.model_params = model_params
        self.best_model_path = best_model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.use_amp = use_amp

    def train_with_masking(self, new_input_to_h1_masking):
        model_params_with_new_masking = self.model_params.copy()
        model_params_with_new_masking['input_to_h1_masking'] = new_input_to_h1_masking

        model_trainer = ModelTrainer(self.model_class, self.dataset, model_params_with_new_masking, self.best_model_path, 
                                     k_folds=5, epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                                     patience=44, device=self.device, use_amp=self.use_amp, save_model=False)
        # Adjusted unpacking to match the returned values
        best_model_state_dict, _, _, _, _, _, _, _, _, _, _ = model_trainer.train(main_logger, verbose=False)

        return best_model_state_dict

    def run_permutation_test(self, K, gene_pathway_matrix, h1_to_pathway_masking, target_class=1):
        pathway_dim = self.model_params['pathway_dim']
        all_lrp_scores = np.zeros((K, pathway_dim))

        for k in range(K):
            main_logger.info(f"Permutation test iteration {k+1}/{K}")
            shuffled_input_to_h1_masking = MatrixHandler.create_shuffled_input_to_h1_masking(gene_pathway_matrix, h1_to_pathway_masking)
            ptest_model_state_dict = self.train_with_masking(shuffled_input_to_h1_masking)

            model = self.model_class(**self.model_params).to(self.device)
            model.load_state_dict(ptest_model_state_dict)

            avg_lrp_scores = self.compute_lrp_scores(model, self.dataset, target_class)
            all_lrp_scores[k, :] = avg_lrp_scores

        return all_lrp_scores

    def compute_lrp_scores(self, model, dataset, target_class=1, num_iterations=100):
        main_logger.info("Computing LRP scores...")
        return LRPHandler.compute_lrp_for_dataset_multiple_times(model, dataset, num_iterations=num_iterations, target_class=target_class)

    def compute_p_value(self, actual_scores, permuted_scores):
        actual_scores = actual_scores.flatten()
        permuted_scores = np.array(permuted_scores).reshape(len(permuted_scores), -1)
        p_values = np.mean(permuted_scores >= actual_scores[:, None], axis=1)
        return p_values

    def get_p_values_for_all_pathways(self, lrp_scores, permuted_scores):
        p_values = self.compute_p_value(lrp_scores, permuted_scores.T)
        return p_values

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
        relevance_output = torch.zeros_like(output).to(device)
        relevance_output[:, target_class] = output[:, target_class].clamp(min=epsilon)  # Clamp the output to a small value

        relevance_hidden_two = LRPHandler.lrp_linear(x_hidden_two, output, model.output_layer.weight, model.output_layer.bias, relevance_output)
        relevance_pathway_layer = LRPHandler.lrp_linear(x_pathway_layer, x_hidden_two, model.hidden_two.weight, model.hidden_two.bias, relevance_hidden_two)
        relevance_hidden_one = LRPHandler.lrp_linear(x_hidden_one, x_pathway_layer, model.pathway_layer.weight, model.pathway_layer.bias, relevance_pathway_layer)
        return relevance_pathway_layer

    @staticmethod
    def compute_lrp_for_dataset_multiple_times(model, dataset, batch_size=32, num_iterations=1000, target_class=1):
        pathway_dim = model.pathway_layer.out_features
        all_lrp_scores = np.zeros((num_iterations, pathway_dim))

        for iteration in tqdm(range(num_iterations), desc="Computing LRP scores", leave=False):
            iteration_lrp_scores = np.zeros(pathway_dim)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
    def print_top_lrp_scores(logger, top_pathways):
        logger.info(f"Top {len(top_pathways)} LRP scores and corresponding pathways:")
        logger.info("----------------------------------------")
        for pathway_name, score, gene_count, p_value in top_pathways:
            logger.info(f"Pathway: {pathway_name}, LRP Score: {score}, Number of Genes: {gene_count}, p-value: {p_value}")

    @staticmethod
    def print_p_values(logger, p_values, pathway_list, lrp_scores, save_path):
        logger.info(f"p-values for the pathways:")
        logger.info("----------------------------------------")
        pathway_data = []
        for pathway_name, lrp_score, p_value in zip(pathway_list, lrp_scores, p_values):
            logger.info(f"Pathway Name: {pathway_name}, LRP Score: {lrp_score}, p-value: {p_value}")
            pathway_data.append((pathway_name, lrp_score, p_value))
        
        df_pathways = pd.DataFrame(pathway_data, columns=['Pathway_Name', 'LRP_Score', 'p_value'])
        df_pathways.to_csv(save_path, index=False)
        logger.info(f"Pathway data saved to {save_path}")

if __name__ == "__main__":
    config = get_config()
    main_logger.info("Loading data...")
    df_filtered_gene_expression, df_filtered_gene_pathway = DataProcessor.load_data(config)
    main_logger.info("Data loaded successfully.")

    main_logger.info("Processing data...")
    gene_list, gene_count, sorted_pathways, hidden_one_list = DataProcessor.process_data(
        df_filtered_gene_expression, df_filtered_gene_pathway, 
        config["params"]["min_genes_threshold"], config["params"]["max_genes_threshold"],
        config["params"]["top_30_nodes"], config["params"]["middle_30_nodes"], config["params"]["bottom_40_nodes"]
    )
    main_logger.info("Data processed successfully.")

    main_logger.info(f"Number of genes: {gene_count}")
    main_logger.info(f"Number of pathways: {len(sorted_pathways)}")
    main_logger.info(f"Number of hidden nodes: {len(hidden_one_list)}")

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
    hidden_two_dim = 128  # Example value
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

    main_logger.info("Initializing KFoldTraining for regular training...")
    kfold_training = ModelTrainer(EXPNet, dataset, model_params, save_model_path, k_folds=config["training"]["k_folds"], 
                                    epochs=200, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4,
                                    patience=config["training"]["patience"], device=device, use_amp=True, save_model=True)

    main_logger.info("Starting training with model saving enabled...")
    best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision = kfold_training.train(main_logger)

    main_logger.info("Running initial LRP computation...")
    kFold_best_model = LRPNet(**model_params).to(device)
    kFold_best_model.load_state_dict(best_model_state_dict)
    obtained_lrp_scores = LRPHandler.compute_lrp_for_dataset_multiple_times(kFold_best_model, dataset, target_class=1)

    start = time.time()

    main_logger.info("Running permutation test...")
    ptest_trainer = p_Test(LRPNet, dataset, model_params, save_model_path, epochs=1000, batch_size=config["training"]["batch_size"], learning_rate=0.001, weight_decay=3e-4, device=device, use_amp=True)
    K = config["params"]["num_permutation_iterations"]
    all_lrp_scores = ptest_trainer.run_permutation_test(K, gene_pathway_matrix, h1_to_pathway_masking, target_class=1)
    
    main_logger.info("Computing p-values...")
    p_values = ptest_trainer.get_p_values_for_all_pathways(obtained_lrp_scores, all_lrp_scores)

    top_pathways, top_indices = LRPHandler.get_top_lrp_scores(obtained_lrp_scores, sorted_pathways, df_filtered_gene_pathway, top_n=50)
    top_pathways_with_p_values = [(pathway_name, score, gene_count, p_values[idx]) for (pathway_name, score, gene_count), idx in zip(top_pathways, top_indices)]

    # Adjusted to include avg_avg_precision and best_avg_precision
    ResultPrinter.print_training_results(main_logger, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision)
    ResultPrinter.print_top_lrp_scores(main_logger, top_pathways_with_p_values)
    csv_save_path = config["file_paths"]["result_save"]
    ResultPrinter.print_p_values(main_logger, p_values, sorted_pathways, obtained_lrp_scores, csv_save_path)

    main_logger.info("Permutation test and p-value computation completed.")
    end = time.time()
    print(f"pTEST time... {end - start:.5f} sec")
