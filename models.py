import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 1)

    def forward(self, data):
        return self.linear(data)  # returns a logit


class CreditRiskMLP(nn.Module):
    def __init__(self, num_features=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, data):
        return self.network(data)  # returns a logit