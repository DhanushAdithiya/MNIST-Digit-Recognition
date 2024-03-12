import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(in_channels, out_channels[0]),
            nn.ReLU(),
            nn.Linear(out_channels[0], out_channels[1]),
            nn.ReLU(),
            nn.Linear(out_channels[1], 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.conv(x)
