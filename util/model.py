import torch
import torch.nn as nn

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

class LRPNet(EXPNet):
    def forward(self, x):
        x_hidden_one = self.leaky_relu(self.hidden_one(x))
        x_hidden_one_drop = self.dropout1(x_hidden_one)
        x_pathway_layer = self.leaky_relu(self.pathway_layer(x_hidden_one_drop))
        x_pathway_layer_drop = self.dropout2(x_pathway_layer)
        x_hidden_two = self.leaky_relu(self.hidden_two(x_pathway_layer_drop))
        x_output = self.output_layer(x_hidden_two)
        return x_output, x_hidden_one, x_pathway_layer, x_hidden_two