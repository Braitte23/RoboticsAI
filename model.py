import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

class NavigationNet(nn.Module):
    def __init__(self, input_features, num_classes):
        super(NavigationNet, self).__init__()
        self.fc1 = nn.Linear(input_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cargar el escalador guardado en el entrenamiento
scaler = joblib.load("scaler.pkl")

def scale_input(x):
    return scaler.transform(x)

