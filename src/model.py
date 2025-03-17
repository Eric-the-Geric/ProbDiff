import torch
import torch.nn as nn
class SimpleUNet(nn.Module):
    """A smaller U-Net to predict noise"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Fewer channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Decoder
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Output has 1 channel
        
    def forward(self, x, t):
        # Encoder
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Decoder
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)  # No activation for the final layer
        
        return x
