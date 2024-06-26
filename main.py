import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from DCLS.construct.modules import Dcls2d
import logging

logging.basicConfig(level=logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self, version):
        super(SimpleCNN, self).__init__()
        self.conv1 = Dcls2d(1, 32, kernel_count=16, dilated_kernel_size=5, padding=2, version=version)
        self.conv2 = Dcls2d(32, 64, kernel_count=32, dilated_kernel_size=5, padding=2, version=version)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

criterion = nn.CrossEntropyLoss()

def train(net, optimizer):
    for epoch in range(5):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

def evaluate(net):
    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    print(classification_report(y_true, y_pred, digits=4))

if __name__ == '__main__':
    versions = ['v0_ste', 'v0', 'v1', 'max', 'gauss']
    for version in versions:
        print(version)
        print('-' * 80)
        net = SimpleCNN(version).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train(net, optimizer)
        evaluate(net)
        print('=' * 80)
        print()
