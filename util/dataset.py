import torch
from torch.utils.data import Dataset

class loadEXPdataset(Dataset):
    def __init__(self, features, labels, device=torch.device("cpu")):
        self.X_tensor = torch.Tensor(features).to(device)
        self.y_tensor = torch.Tensor(labels).long().to(device)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, index):
        return {'features': self.X_tensor[index], 'labels': self.y_tensor[index]}