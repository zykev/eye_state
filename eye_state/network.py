import torch.nn as nn
import torch
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLu(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLu(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Relu(),
            nn.Linear(in_features=84, out_features=1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        output = self.linear(x)
        return output

