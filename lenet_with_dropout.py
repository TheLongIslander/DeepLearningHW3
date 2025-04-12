import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import platform
import os
import sys
from contextlib import contextmanager

# Transform for CIFAR-10 (normalize RGB channels)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# CIFAR-10 dataset loading
with suppress_stdout():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)

# LeNet-5 with reduced Dropout
class LeNet5Dropout(nn.Module):
    def __init__(self):
        super(LeNet5Dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.dropout_conv = nn.Dropout(p=0.1)       # ↓ reduced
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout_fc1 = nn.Dropout(p=0.3)        # ↓ reduced
        self.fc2 = nn.Linear(120, 84)
        self.dropout_fc2 = nn.Dropout(p=0.3)        # ↓ reduced
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout_fc1(F.relu(self.fc1(x)))
        x = self.dropout_fc2(F.relu(self.fc2(x)))
        return self.fc3(x)

# Train and evaluate function
def train_and_evaluate(model, trainloader, testloader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    start_time = time.time()
    model.train()
    for epoch in range(20):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")
    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Training Time: {train_time:.2f} seconds")
    return accuracy, train_time

# Run
if __name__ == "__main__":
    if torch.backends.mps.is_available() and platform.processor() == "arm":
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    model = LeNet5Dropout()
    train_and_evaluate(model, trainloader, testloader, device)
