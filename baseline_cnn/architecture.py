import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    """LeNet-style: two 5x5 conv blocks and two dense layers (about 65k parameters)."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                               nn.Linear(32 * 5 * 5, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.f(x)
