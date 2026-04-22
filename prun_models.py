import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neural_compressor.config import WeightPruningConfig
try:
    # สำหรับ Neural Compressor 2.0 及更高版本的 API
    from neural_compressor.training import prepare_compression
except ImportError:
    # 兼容老版本 API
    from neural_compressor.training import prepare_pruning as prepare_compression

from model_definitions import SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for pruning: {device}")

mnist_dir = os.path.join('models_', 'MNIST')
cifar_dir = os.path.join('models_', 'CIFAR-10')

original_models_mnist = {
    "SimpleCNN": SimpleCNN(),
    "LeNet-4": LeNet4(),
    "LeNet-5": LeNet5()
}

original_models_cifar = {
    "PlainNet-20": PlainNet20(),
    "ResNet-20": ResNet20(),
    "VGG-16": VGG16()
}

# General Pruning Config 
# Here we use magnitude-based unstructured pruning as an example
prune_config = WeightPruningConfig(
    pruning_type="magnitude",
    target_sparsity=0.5, # 50% sparsity
    pattern="1x1"
)

# ====================================================================
# MNIST Pruning
# ====================================================================
final_pruning_accuracies = {}

print("\n--- Processing MNIST Pruning ---")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=1000, shuffle=False)

for model_name, model in original_models_mnist.items():
    model_path = os.path.join(mnist_dir, f'{model_name.lower()}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nPruning {model_name} -> {model_name}_prun ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    compression_manager = prepare_compression(model, prune_config)
    
    # 模拟一次训练过程来让剪枝生效
    compression_manager.callbacks.on_train_begin()
    compression_manager.callbacks.on_epoch_begin(0)
    compression_manager.callbacks.on_step_begin(0)
    compression_manager.callbacks.on_step_end()
    compression_manager.callbacks.on_epoch_end()
    compression_manager.callbacks.on_train_end()
    
    pruned_model = compression_manager.model
    
    # Evaluate pruned model
    pruned_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in mnist_test_loader:
            data, target = data.to(device), target.to(device)
            outputs = pruned_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    final_pruning_accuracies[f"{model_name}_prun"] = accuracy
    print(f"Pruned Model Accuracy of {model_name} on test set: {accuracy:.2f}%")
    
    save_path = os.path.join(mnist_dir, f'{model_name.lower()}_prun.pth')
    torch.save(pruned_model.state_dict(), save_path)
    print(f"Saved pruned model to {save_path}")

# ====================================================================
# CIFAR-10 Pruning
# ====================================================================
print("\n--- Processing CIFAR-10 Pruning ---")
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=128, shuffle=True)
cifar_test_loader = DataLoader(cifar_test_dataset, batch_size=100, shuffle=False)

for model_name, model in original_models_cifar.items():
    saved_name = model_name.lower().replace("-", "")
    model_path = os.path.join(cifar_dir, f'{saved_name}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nPruning {model_name} -> {model_name}_prun ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    compression_manager = prepare_compression(model, prune_config)
    
    compression_manager.callbacks.on_train_begin()
    compression_manager.callbacks.on_epoch_begin(0)
    compression_manager.callbacks.on_step_begin(0)
    compression_manager.callbacks.on_step_end()
    compression_manager.callbacks.on_epoch_end()
    compression_manager.callbacks.on_train_end()
    
    pruned_model = compression_manager.model
    
    # Evaluate pruned model
    pruned_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in cifar_test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = pruned_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total
    final_pruning_accuracies[f"{model_name}_prun"] = accuracy
    print(f"Pruned Model Accuracy of {model_name} on test set: {accuracy:.2f}%")
    
    save_path = os.path.join(cifar_dir, f'{saved_name}_prun.pth')
    torch.save(pruned_model.state_dict(), save_path)
    print(f"Saved pruned model to {save_path}")

print("\n" + "="*50)
print("FINAL PRUNED MODEL ACCURACIES")
print("="*50)
for model_name, acc in final_pruning_accuracies.items():
    print(f"{model_name:<15}: {acc:.2f}%")
print("="*50)
print("Pruning process completed using Neural Compressor.")
print("="*50)
