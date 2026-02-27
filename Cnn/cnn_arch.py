import torch
import torch.nn as nn
import torch.nn.functional as F

# Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 1st Convolution Layer
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB image
            out_channels=16,    # 16 filters
            kernel_size=3,      # 3x3 kernel
            stride=1,
            padding=1           # Keep same size
        )
        
        # 2nd Convolution Layer
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        
        # Conv1 → ReLU → Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)   # 32x32 → 16x16
        
        # Conv2 → ReLU → Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)   # 16x16 → 8x8
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# Create Model
model = SimpleCNN()
print(model)