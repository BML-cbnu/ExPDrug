import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score
from tqdm import tqdm
import numpy as np
import pandas as pd
from .model import FocalLoss, LRPNet
from .logger import Logger, ResultPrinter
from .data_processor import MatrixHandler

class ModelTrainer:
    """
    ModelTrainer class for training the neural network model.
    """
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
        """
        Train the model using cross-validation and return the best model's state dict and performance metrics.
        """
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True)
        fold_results = []
        fold_model_state_dicts = []

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
            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            best_val_loss = float('inf')
            epochs_no_improve = 0

            best_accuracy = 0
            best_roc_auc = 0
            best_precision = 0
            best_f1 = 0
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
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.cuda.amp.autocast(enabled=self.use_amp):
                            outputs = model(batch['features'])
                            if isinstance(outputs, tuple):
                                outputs = outputs[0]
                            loss = criterion(outputs, batch['labels'])
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        val_labels.extend(batch['labels'].cpu().numpy())
                        val_preds.extend(preds.cpu().numpy())
                
                val_loss /= len(val_loader)

                accuracy = accuracy_score(val_labels, val_preds)
                roc_auc = roc_auc_score(val_labels, val_preds, average='weighted', multi_class='ovr')
                precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
                f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                if precision > best_precision:
                    best_precision = precision
                if f1 > best_f1:
                    best_f1 = f1

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        early_stopping_epoch = epoch + 1
                        break

            fold_results.append((best_accuracy, best_roc_auc, best_precision, best_f1))
            fold_model_state_dicts.append(model.state_dict())

        avg_accuracy = np.mean([result[0] for result in fold_results])
        avg_roc_auc = np.mean([result[1] for result in fold_results])
        avg_precision = np.mean([result[2] for result in fold_results])
        avg_f1 = np.mean([result[3] for result in fold_results])

        best_accuracy = np.max([result[0] for result in fold_results])
        best_roc_auc = np.max([result[1] for result in fold_results])
        best_precision = np.max([result[2] for result in fold_results])
        best_f1 = np.max([result[3] for result in fold_results])

        closest_fold_idx = np.argmin([np.abs(result[1] - avg_roc_auc) for result in fold_results])
        best_model_state_dict = fold_model_state_dicts[closest_fold_idx]

        if self.save_model and best_model_state_dict is not None:
            torch.save(best_model_state_dict, self.save_model_path)
            logger.info(f"Saved model closest to average performance to {self.save_model_path}")

        return best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, best_accuracy, best_roc_auc, best_precision, best_f1

class p_Test:
    """
    p_Test class for running permutation tests.
    """
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
        """
        Train the model with a new masking matrix.
        """
        model_params_with_new_masking = self.model_params.copy()
        model_params_with_new_masking['input_to_h1_masking'] = new_input_to_h1_masking

        model_trainer = ModelTrainer(self.model_class, self.dataset, model_params_with_new_masking, self.best_model_path, 
                                     k_folds=5, epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                                     patience=44, device=self.device, use_amp=self.use_amp, save_model=False)
        best_model_state_dict, _, _, _, _, _, _, _, _ = model_trainer.train(Logger(), verbose=False)

        return best_model_state_dict

    def run_permutation_test(self, K, gene_pathway_matrix, h1_to_pathway_masking, target_class=1):
        """
        Run permutation tests to compute LRP scores.
        """
        pathway_dim = self.model_params['pathway_dim']
        all_lrp_scores = np.zeros((K, pathway_dim))

        for k in range(K):
            Logger().info(f"Permutation test iteration {k+1}/{K}")
            shuffled_input_to_h1_masking = MatrixHandler.create_shuffled_input_to_h1_masking(gene_pathway_matrix, h1_to_pathway_masking)
            ptest_model_state_dict = self.train_with_masking(shuffled_input_to_h1_masking)

            model = self.model_class(**self.model_params).to(self.device)
            model.load_state_dict(ptest_model_state_dict)

            avg_lrp_scores = self.compute_lrp_scores(model, self.dataset, target_class)
            all_lrp_scores[k, :] = avg_lrp_scores

        return all_lrp_scores

    def compute_lrp_scores(self, model, dataset, target_class=1, num_iterations=100):
        """
        Compute LRP scores for the dataset.
        """
        Logger().info("Computing LRP scores...")
        return LRPHandler.compute_lrp_for_dataset_multiple_times(model, dataset, num_iterations=num_iterations, target_class=target_class)

    def compute_p_value(self, actual_scores, permuted_scores):
        """
        Compute p-values based on actual and permuted LRP scores.
        """
        actual_scores = actual_scores.flatten()
        permuted_scores = np.array(permuted_scores).reshape(len(permuted_scores), -1)
        p_values = np.mean(permuted_scores >= actual_scores[:, None], axis=1)
        return p_values

    def get_p_values_for_all_pathways(self, lrp_scores, permuted_scores):
        """
        Get p-values for all pathways.
        """
        p_values = self.compute_p_value(lrp_scores, permuted_scores.T)
        return p_values

class LRPHandler:
    """
    LRPHandler class for handling LRP computations.
    """
    @staticmethod
    def lrp_linear(h_in, h_out, weight, bias, relevance_out, epsilon=1e-6):
        """
        Compute LRP relevance for a linear layer.
        """
        z = h_in @ weight.t() + bias
        s = relevance_out / (z + epsilon)
        c = s @ weight
        relevance_in = h_in * c
        return relevance_in

    @staticmethod
    def compute_lrp(model, x, target_class, epsilon=1e-6):
        """
        Compute LRP relevance scores for the model.
        """
        model.eval()
        output, x_hidden_one, x_pathway_layer, x_hidden_two = model(x)
        relevance_output = torch.zeros_like(output).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        relevance_output[:, target_class] = output[:, target_class].clamp(min=epsilon)

        relevance_hidden_two = LRPHandler.lrp_linear(x_hidden_two, output, model.output_layer.weight, model.output_layer.bias, relevance_output)
        relevance_pathway_layer = LRPHandler.lrp_linear(x_pathway_layer, x_hidden_two, model.hidden_two.weight, model.hidden_two.bias, relevance_hidden_two)
        relevance_hidden_one = LRPHandler.lrp_linear(x_hidden_one, x_pathway_layer, model.pathway_layer.weight, model.pathway_layer.bias, relevance_pathway_layer)
        return relevance_pathway_layer

    @staticmethod
    def compute_lrp_for_dataset_multiple_times(model, dataset, batch_size=32, num_iterations=1000, target_class=1):
        """
        Compute LRP relevance scores for the dataset multiple times.
        """
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
        """
        Get top LRP scores and corresponding pathways.
        """
        lrp_scores = lrp_scores.flatten()
        top_indices = np.argsort(lrp_scores)[-top_n:][::-1]
        top_pathways = []
        for idx in top_indices:
            pathway_name = pathway_list[idx]
            score = lrp_scores[idx]
            num_genes = len(gene_pathway_df[gene_pathway_df['pathway_name'] == pathway_name])
            top_pathways.append((pathway_name, score, num_genes))
        return top_pathways, top_indices
