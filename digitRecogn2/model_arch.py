import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGeoModel(nn.Module):
    def __init__(self, num_geo_features=7, num_classes=10):
        super(CNNGeoModel, self).__init__()
        # Convolutional blocks with pooling after each conv
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened dimension: 28->26->13->11->5->3->1
        self.flatten_dim = 128 * 1 * 1

        # Fully connected layers including geometric features
        self.fc1 = nn.Linear(self.flatten_dim + num_geo_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_img, x_geo):
        # Pass through conv + pooling blocks
        x = F.relu(self.conv1(x_img))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Concatenate geometric features
        x = torch.cat([x, x_geo], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out