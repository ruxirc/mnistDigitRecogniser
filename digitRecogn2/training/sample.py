import os

#%% md
# # This is a sample Jupyter Notebook
#
# Below is an example of a code cell.
# Put your cursor into the cell and press Shift+Enter to execute it and select the next one, or click 'Run Cell' button.
#
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
# To learn more about Jupyter Notebooks in PyCharm, see [help](https://www.jetbrains.com/help/pycharm/ipython-notebook-support.html).
# For an overview of PyCharm, go to Help -> Learn IDE features or refer to [our documentation](https://www.jetbrains.com/help/pycharm/getting-started.html).
#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np

# --- Load MNIST with normalization ---
transform = transforms.Compose([
    transforms.ToTensor(),                     # transforms to [0,1], shape (1,28,28)
    transforms.Normalize((0.1307,), (0.3081,)) # normalize to mean/std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Extract image arrays and labels
x_train = train_dataset.data.numpy().astype('float32')
y_train = train_dataset.targets.numpy()
x_test  = test_dataset.data.numpy().astype('float32')
y_test  = test_dataset.targets.numpy()

# Normalize pixel values to [0,1]
x_train /= 255.0
x_test  /= 255.0

# Apply dataset normalization (mean/std)
mean, std = 0.1307, 0.3081
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std
#%%
from digitRecogn2.utils.geometric_props import calculate_geometric_features

# Compute geometric features for datasets
geo_train = np.array([calculate_geometric_features(img) for img in x_train])
geo_test  = np.array([calculate_geometric_features(img) for img in x_test])

# Convert everything to torch tensors
x_train_tensor   = torch.tensor(x_train).unsqueeze(1)        # (N,1,28,28)
geo_train_tensor = torch.tensor(geo_train, dtype=torch.float32)
y_train_tensor   = torch.tensor(y_train, dtype=torch.long)

x_test_tensor    = torch.tensor(x_test).unsqueeze(1)
geo_test_tensor  = torch.tensor(geo_test, dtype=torch.float32)
y_test_tensor    = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_loader = DataLoader(TensorDataset(x_train_tensor, geo_train_tensor, y_train_tensor),
                          batch_size=64, shuffle=True)
test_loader  = DataLoader(TensorDataset(x_test_tensor,  geo_test_tensor,  y_test_tensor),
                          batch_size=64, shuffle=False)
#%%
from digitRecogn2.modelArchs.model_arch import CNNGeoModel

def train_model(model, train_loader, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct      = 0
        total        = 0
        for imgs, geos, labels in train_loader:
            imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, geos)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - "
              f"Accuracy: {100*correct/total:.2f}%")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNNGeoModel(num_classes=10).to(device)

# Train model
train_model(model, train_loader, epochs=50)

# Save trained model
os.makedirs('../models3', exist_ok=True)
torch.save(model.state_dict(), '../models3/m1.pth')


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, geos, labels in test_loader:
            imgs, geos, labels = imgs.to(device), geos.to(device), labels.to(device)
            outputs = model(imgs, geos)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

# Load and evaluate
model.load_state_dict(torch.load('../models3/m1.pth'))
test_model(model, test_loader)
