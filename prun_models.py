import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neural_compressor.config import WeightPruningConfig
from neural_compressor.training import prepare_pruning, WeightPruning

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
    pattern="4x1"
)

# ====================================================================
# MNIST Pruning
# ====================================================================
print("\n--- Processing MNIST Pruning ---")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)

for model_name, model in original_models_mnist.items():
    model_path = os.path.join(mnist_dir, f'{model_name.lower()}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nPruning {model_name} -> {model_name}_prun ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Initialize Pruning object
    pruning = WeightPruning(prune_config)
    pruning.model = model
    
    # For magnitude pruning we normally just need to run pruning setup 
    pruning.pre_epoch_begin()
    pruning.on_step_begin(0)
    pruning.on_step_end()
    pruning.on_epoch_end()
    
    pruned_model = pruning.model
    
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
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=128, shuffle=True)

for model_name, model in original_models_cifar.items():
    saved_name = model_name.lower().replace("-", "")
    model_path = os.path.join(cifar_dir, f'{saved_name}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nPruning {model_name} -> {model_name}_prun ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    pruning = WeightPruning(prune_config)
    pruning.model = model
    
    pruning.pre_epoch_begin()
    pruning.on_step_begin(0)
    pruning.on_step_end()
    pruning.on_epoch_end()
    
    pruned_model = pruning.model
    
    save_path = os.path.join(cifar_dir, f'{saved_name}_prun.pth')
    torch.save(pruned_model.state_dict(), save_path)
    print(f"Saved pruned model to {save_path}")

print("\n" + "="*50)
print("Pruning process completed using Neural Compressor.")
print("="*50)
