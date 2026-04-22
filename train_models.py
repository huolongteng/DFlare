
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import model definitions from the previously created file
from model_definitions import SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16

print("Step 1: Create directories for models.")
# Create base directory
base_dir = 'models_'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Create subdirectories for datasets
mnist_dir = os.path.join(base_dir, 'MNIST')
cifar_dir = os.path.join(base_dir, 'CIFAR-10')
if not os.path.exists(mnist_dir):
    os.makedirs(mnist_dir)
if not os.path.exists(cifar_dir):
    os.makedirs(cifar_dir)
print(f"Directories '{mnist_dir}' and '{cifar_dir}' created successfully.")
print("-" * 50)

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Step 2: Set device to '{device}'.")
print("-" * 50)

# Dictionary to collect all final accuracies
final_accuracies = {}

# --- MNIST Models Training and Evaluation ---
print("Step 3: Handle MNIST models (SimpleCNN, LeNet-4, LeNet-5).")

# Data loading
print("Loading MNIST dataset...")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=1000, shuffle=False)
print("MNIST dataset loaded.")

# List of MNIST models to train
mnist_models_to_train = {
    "SimpleCNN": SimpleCNN(),
    "LeNet-4": LeNet4(),
    "LeNet-5": LeNet5()
}

# Training and evaluation loop for MNIST
for model_name, model in mnist_models_to_train.items():
    print(f"\n--- Training {model_name} on MNIST ---")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simple training loop
    num_epochs = 3 # Using fewer epochs for a quick run
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(mnist_train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(mnist_train_loader)}], Loss: {loss.item():.4f}")

    print(f"--- Evaluating {model_name} on MNIST test set ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in mnist_test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    final_accuracies[model_name] = accuracy
    
    accuracy = 100 * correct / total
    print(f"Accuracy of {model_name} on the test set: {accuracy:.2f}%")

    # Save the model
    model_path = os.path.join(mnist_dir, f'{model_name.lower()}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model {model_name} saved to {model_path}")

print("-" * 50)

# --- CIFAR-10 Models Training and Evaluation ---
print("Step 4: Handle CIFAR-10 models (PlainNet-20, ResNet-20, VGG-16).")

# Data loading
print("Loading CIFAR-10 dataset...")
transform_cifar = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=128, shuffle=True)
cifar_test_loader = DataLoader(cifar_test_dataset, batch_size=100, shuffle=False)
print("CIFAR-10 dataset loaded.")

# List of CIFAR-10 models to train
cifar_models_to_train = {
    "PlainNet-20": PlainNet20(),
    "ResNet-20": ResNet20(),
    "VGG-16": VGG16()
}

# Training and evaluation loop for CIFAR-10
for model_name, model in cifar_models_to_train.items():
    print(f"\n--- Training {model_name} on CIFAR-10 ---")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 针对 PlainNet-20 和 ResNet-20 提升超参数 (增加 epoch，使用更大的初始学习率 0.1)
    if model_name in ["PlainNet-20", "ResNet-20"]:
        lr = 0.1
        num_epochs = 100
    else:
        lr = 0.01
        num_epochs = 10
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Simple training loop
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(cifar_train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(cifar_train_loader)}], Loss: {loss.item():.4f}")
        scheduler.step()

    print(f"--- Evaluating {model_name} on CIFAR-10 test set ---")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in cifar_test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    final_accuracies[model_name] = accuracy
    print(f"Accuracy of {model_name} on the test set: {accuracy:.2f}%")

    # Save the model
    model_path = os.path.join(cifar_dir, f'{model_name.lower().replace("-", "")}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model {model_name} saved to {model_path}")


print("\n" + "="*50)
print("FINAL MODEL ACCURACIES SUMMARY")
print("="*50)
for model_name, acc in final_accuracies.items():
    print(f"{model_name:<15}: {acc:.2f}%")
print("="*50)
print("-" * 50)
print("Step 5: All models have been trained, evaluated, and saved.")
