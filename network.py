import torch.nn as nn
import torch
from torchsummary import summary

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features=144, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        output = self.linear(x)
        return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    summary(model,(1,24,24))