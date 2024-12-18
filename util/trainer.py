import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch import nn

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
    def __init__(self, model_class, dataset, model_params, save_model_path, k_folds=5, epochs=200, 
                 batch_size=64, learning_rate=0.001, weight_decay=3e-4, patience=44, 
                 device=torch.device("cpu"), use_amp=True, save_model=True):
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

        roc_auc_scores = [result[1] for result in fold_results]
        sorted_indices = np.argsort(roc_auc_scores)
        median_index = sorted_indices[len(sorted_indices) // 2]
        best_model_state_dict = fold_model_state_dicts[median_index]

        if self.save_model and best_model_state_dict is not None:
            torch.save(best_model_state_dict, self.save_model_path)
            logger.info(f"Saved median model from fold {median_index + 1} to {self.save_model_path}")

        return best_model_state_dict, avg_accuracy, avg_roc_auc, avg_precision, avg_f1, avg_avg_precision, best_accuracy, best_roc_auc, best_precision, best_f1, best_avg_precision