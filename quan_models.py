import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neural_compressor.config import PostTrainingQuantConfig
from neural_compressor.quantization import fit

from model_definitions import SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16

# === General Settings ===
# For INC Post-Training Quantization, CPU is recommended and usually required for the inference backend
device = torch.device("cpu")
print(f"Using device for quantization: {device}")

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

# ====================================================================
# MNIST Quantization
# ====================================================================
print("\n--- Processing MNIST Quantization ---")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
# A small batch size for calibration
mnist_calib_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)

conf = PostTrainingQuantConfig()

for model_name, model in original_models_mnist.items():
    model_path = os.path.join(mnist_dir, f'{model_name.lower()}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nQuantizing {model_name} -> {model_name}_quan ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Perform Post-Training Quantization (PTQ)
    q_model = fit(model=model, conf=conf, calib_dataloader=mnist_calib_loader)
    
    # Save the quantized model
    save_path = os.path.join(mnist_dir, f'{model_name.lower()}_quan')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    q_model.save(save_path)
    print(f"Saved quantized model to {save_path}")

# ====================================================================
# CIFAR-10 Quantization
# ====================================================================
print("\n--- Processing CIFAR-10 Quantization ---")
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
cifar_calib_loader = DataLoader(cifar_test_dataset, batch_size=64, shuffle=False)

for model_name, model in original_models_cifar.items():
    saved_name = model_name.lower().replace("-", "")
    model_path = os.path.join(cifar_dir, f'{saved_name}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nQuantizing {model_name} -> {model_name}_quan ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Perform Post-Training Quantization (PTQ)
    q_model = fit(model=model, conf=conf, calib_dataloader=cifar_calib_loader)
    
    # Save the quantized model
    save_path = os.path.join(cifar_dir, f'{saved_name}_quan')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    q_model.save(save_path)
    print(f"Saved quantized model to {save_path}")

print("\n" + "="*50)
print("Quantization process completed using Neural Compressor.")
print("="*50)
