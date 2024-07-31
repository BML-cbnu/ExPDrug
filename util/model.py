import torch.nn as nn
import torch

class EXPNet(nn.Module):
    """
    EXPNet class for defining the neural network architecture.
    """
    def __init__(self, input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1=0.8, dropout_rate2=0.7):
        super(EXPNet, self).__init__()
        
        # Define layers
        self.hidden_one = nn.Linear(input_dim, hidden_one_dim, bias=True)
        self.pathway_layer = nn.Linear(hidden_one_dim, pathway_dim, bias=True)
        self.hidden_two = nn.Linear(pathway_dim, hidden_two_dim, bias=True)
        self.output_layer = nn.Linear(hidden_two_dim, output_dim, bias=True)
        
        # Activation functions and dropout
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.dropout2 = nn.Dropout(dropout_rate2)

        # Convert masking matrices to tensors and apply to weights
        input_to_h1_masking_tensor = torch.Tensor(input_to_h1_masking.T)
        h1_to_pathway_masking_tensor = torch.Tensor(h1_to_pathway_masking.T)
        
        with torch.no_grad():
            self.hidden_one.weight *= input_to_h1_masking_tensor
            self.pathway_layer.weight *= h1_to_pathway_masking_tensor

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.leaky_relu(self.hidden_one(x))
        x = self.dropout1(x)
        x = self.leaky_relu(self.pathway_layer(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.hidden_two(x))
        x = self.output_layer(x)
        return x

class LRPNet(EXPNet):
    """
    LRPNet class extending EXPNet for Layer-wise Relevance Propagation.
    """
    def __init__(self, input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1=0.8, dropout_rate2=0.7):
        super(LRPNet, self).__init__(input_dim, hidden_one_dim, pathway_dim, hidden_two_dim, output_dim, input_to_h1_masking, h1_to_pathway_masking, dropout_rate1, dropout_rate2)
    
    def forward(self, x):
        """
        Forward pass through the network, returning intermediate activations.
        """
        x_hidden_one = self.leaky_relu(self.hidden_one(x))
        x_hidden_one = self.dropout1(x_hidden_one)
        x_pathway_layer = self.leaky_relu(self.pathway_layer(x_hidden_one))
        x_pathway_layer = self.dropout2(x_pathway_layer)
        x_hidden_two = self.leaky_relu(self.hidden_two(x_pathway_layer))
        x_output = self.output_layer(x_hidden_two)
        return x_output, x_hidden_one, x_pathway_layer, x_hidden_two

class FocalLoss(nn.Module):
    """
    FocalLoss class for defining the Focal Loss function.
    """
    def __init__(self, gamma=2, alpha=0.4, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.
        """
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
