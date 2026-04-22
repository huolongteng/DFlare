import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from neural_compressor.config import PostTrainingQuantConfig
try:
    from neural_compressor import quantization
    fit_func = quantization.fit
except ImportError:
    from neural_compressor.quantization import fit as fit_func

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


def build_calibration_loader(train_dataset, subset_size=1024, batch_size=64, seed=42):
    """
    Build a deterministic calibration loader from a training-set subset to avoid test-set leakage.
    """
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator)[:subset_size].tolist()
    calib_subset = Subset(train_dataset, indices)
    return DataLoader(calib_subset, batch_size=batch_size, shuffle=False)

# ====================================================================
# MNIST Quantization
# ====================================================================
final_quant_accuracies = {}

print("\n--- Processing MNIST Quantization ---")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
# Calibrate only on a subset of training data; evaluate only on test data.
mnist_calib_loader = build_calibration_loader(mnist_train_dataset, subset_size=1024, batch_size=64, seed=42)
mnist_eval_loader = DataLoader(mnist_test_dataset, batch_size=1000, shuffle=False)

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
    # We use a dummy eval_func because we just want to forcefully quantize the model without early exit rejections
    q_model = fit_func(model=model, conf=conf, calib_dataloader=mnist_calib_loader, eval_func=lambda m: 1.0)
    
    if q_model is None:
        print(f"Quantization failed for {model_name}!")
        continue

    # Save the quantized model
    save_path = os.path.join(mnist_dir, f'{model_name.lower()}_quan.pth')
    torch.save(q_model.model.state_dict(), save_path)
    print(f"Saved quantized model to {save_path}")

    # Evaluate quantized model
    q_model_pytorch = q_model.model
    q_model_pytorch.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in mnist_eval_loader:
            data, target = data.to(device), target.to(device)
            outputs = q_model_pytorch(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    final_quant_accuracies[f"{model_name}_quan"] = accuracy
    print(f"Quantized Model Accuracy of {model_name} on test set: {accuracy:.2f}%")

# ====================================================================
# CIFAR-10 Quantization
# ====================================================================
print("\n--- Processing CIFAR-10 Quantization ---")
transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
cifar_calib_loader = build_calibration_loader(cifar_train_dataset, subset_size=1024, batch_size=64, seed=42)
cifar_eval_loader = DataLoader(cifar_test_dataset, batch_size=100, shuffle=False)

for model_name, model in original_models_cifar.items():
    saved_name = model_name.lower().replace("-", "")
    model_path = os.path.join(cifar_dir, f'{saved_name}.pth')
    if not os.path.exists(model_path):
        print(f"Original model {model_name} not found at {model_path}. Skipping.")
        continue
        
    print(f"\nQuantizing {model_name} -> {model_name}_quan ...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Perform Post-Training Quantization
    q_model = fit_func(model=model, conf=conf, calib_dataloader=cifar_calib_loader, eval_func=lambda m: 1.0)
    
    if q_model is None:
        print(f"Quantization failed for {model_name}!")
        continue

    # Save the quantized model
    save_path = os.path.join(cifar_dir, f'{saved_name}_quan.pth')
    torch.save(q_model.model.state_dict(), save_path)
    print(f"Saved quantized model to {save_path}")

    # Evaluate quantized model
    q_model_pytorch = q_model.model
    q_model_pytorch.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in cifar_eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = q_model_pytorch(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total
    final_quant_accuracies[f"{model_name}_quan"] = accuracy
    print(f"Quantized Model Accuracy of {model_name} on test set: {accuracy:.2f}%")

print("\n" + "="*50)
print("QUANTIZED MODEL ACCURACIES")
print("="*50)
for model_name, acc in final_quant_accuracies.items():
    print(f"{model_name:<15}: {acc:.2f}%")
print("="*50)
print("Quantization process completed using Neural Compressor.")
print("="*50)
