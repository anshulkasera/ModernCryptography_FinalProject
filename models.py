import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, data):
        return torch.sigmoid(self.linear(data))