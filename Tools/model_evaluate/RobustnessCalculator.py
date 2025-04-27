import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 假设的遥感图像数据集路径
train_data_path = 'path/to/train/data'
test_data_path = 'path/to/test/data'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = ImageFolder(root=train_data_path, transform=transform)
test_dataset = ImageFolder(root=test_data_path, transform=transform)

# 添加噪声
def add_noise(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    return image + noise

# 添加异常值
def add_outliers(image, outlier_level=0.5):
    outliers = torch.randn_like(image) * outlier_level
    return torch.cat([image, outliers], dim=0)

# 创建噪声和异常值数据集
test_noisy_dataset = torch.utils.data.TensorDataset(
    torch.stack([add_noise(img) for img, _ in test_dataset]),
    torch.tensor([label for _, label in test_dataset])
)

test_outliers_dataset = torch.utils.data.TensorDataset(
    torch.stack([add_outliers(img) for img, _ in test_dataset]),
    torch.tensor([label for _, label in test_dataset])
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_noisy_loader = DataLoader(test_noisy_dataset, batch_size=32, shuffle=False)
test_outliers_loader = DataLoader(test_outliers_dataset, batch_size=32, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 假设有两个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 2)  # 假设有两个类别

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return model


def evaluate_model(model, test_loader, test_noisy_loader, test_outliers_loader):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # 泛化能力
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        generalization_accuracy = correct / total

        # 抗噪声能力
        correct_noisy = 0
        total_noisy = 0
        for inputs, labels in test_noisy_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_noisy += labels.size(0)
            correct_noisy += (predicted == labels).sum().item()
        noise_robustness_accuracy = correct_noisy / total_noisy

        # 异常值检测
        correct_outliers = 0
        total_outliers = 0
        for inputs, labels in test_outliers_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_outliers += labels.size(0)
            correct_outliers += (predicted == labels).sum().item()
        outlier_detection_accuracy = correct_outliers / total_outliers

    return generalization_accuracy, noise_robustness_accuracy, outlier_detection_accuracy


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和优化器
simple_model = SimpleCNN().to(device)
complex_model = ComplexCNN().to(device)
criterion = nn.CrossEntropyLoss()
simple_optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
complex_optimizer = optim.Adam(complex_model.parameters(), lr=0.001)

# 训练模型
print("Training Simple Model...")
simple_model = train_model(simple_model, train_loader, criterion, simple_optimizer, num_epochs=10)
print("Training Complex Model...")
complex_model = train_model(complex_model, train_loader, criterion, complex_optimizer, num_epochs=10)

# 评估模型
simple_generalization, simple_noise_robustness, simple_outlier_detection = evaluate_model(simple_model, test_loader, test_noisy_loader, test_outliers_loader)
complex_generalization, complex_noise_robustness, complex_outlier_detection = evaluate_model(complex_model, test_loader, test_noisy_loader, test_outliers_loader)

print("Simple Model:")
print(f"Generalization Accuracy: {simple_generalization:.4f}")
print(f"Noise Robustness Accuracy: {simple_noise_robustness:.4f}")
print(f"Outlier Detection Accuracy: {simple_outlier_detection:.4f}")

print("Complex Model:")
print(f"Generalization Accuracy: {complex_generalization:.4f}")
print(f"Noise Robustness Accuracy: {complex_noise_robustness:.4f}")
print(f"Outlier Detection Accuracy: {complex_outlier_detection:.4f}")
