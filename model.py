"""
Federated Learning Model Definition
Simple Feedforward Neural Network for Telco Customer Churn Prediction
"""

import torch
import torch.nn as nn


class ChurnModel(nn.Module):
    """
    Simple Feedforward Neural Network for predicting customer churn.
    
    Architecture:
    - Input layer: variable size (based on features after encoding)
    - Hidden layer 1: 64 neurons + ReLU
    - Hidden layer 2: 32 neurons + ReLU
    - Output layer: 1 neuron (binary classification)
    - Loss: BCEWithLogitsLoss (no sigmoid in final layer)
    """
    
    def __init__(self, input_size):
        super(ChurnModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
