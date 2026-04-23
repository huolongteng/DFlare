from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_definitions import (
    SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16,
    SimpleCNN_Half, LeNet4_Half, LeNet5_Half, PlainNet8, ResNet8, VGG11
)
import torch
import random

import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)

# Data
# MNIST
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=1000, shuffle=False)

# CIFAR-10
transform_cifar_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar_eval)
cifar_test_loader = DataLoader(cifar_test_dataset, batch_size=100, shuffle=False)


# Dict path
model_paths_mnist = {
    "SimpleCNN": "models_/MNIST/simplecnn.pth",
    "SimpleCNN_kd": "models_/MNIST/simplecnn_kd.pth",
    "SimpleCNN_prun": "models_/MNIST/simplecnn_prun.pth",
    "SimpleCNN_quan": "models_/MNIST/simplecnn_quan.pth",
    "LeNet-4": "models_/MNIST/lenet-4.pth",
    "LeNet-4_kd": "models_/MNIST/lenet-4_kd.pth",
    "LeNet-4_prun": "models_/MNIST/lenet-4_prun.pth",
    "LeNet-4_quan": "models_/MNIST/lenet-4_quan.pth",
    "LeNet-5": "models_/MNIST/lenet-5.pth",
    "LeNet-5_kd": "models_/MNIST/lenet-5_kd.pth",
    "LeNet-5_prun": "models_/MNIST/lenet-5_prun.pth",
    "LeNet-5_quan": "models_/MNIST/lenet-5_quan.pth",
}

model_paths_cifar = {
    "PlainNet-20": "models_/CIFAR-10/plainnet20.pth",
    "PlainNet-20_kd": "models_/CIFAR-10/plainnet20_kd.pth",
    "PlainNet-20_prun": "models_/CIFAR-10/plainnet20_prun.pth",
    "PlainNet-20_quan": "models_/CIFAR-10/plainnet20_quan.pth",
    "ResNet-20": "models_/CIFAR-10/resnet20.pth",
    "ResNet-20_kd": "models_/CIFAR-10/resnet20_kd.pth",
    "ResNet-20_prun": "models_/CIFAR-10/resnet20_prun.pth",
    "ResNet-20_quan": "models_/CIFAR-10/resnet20_quan.pth",
    "VGG-16": "models_/CIFAR-10/vgg16.pth",
    "VGG-16_kd": "models_/CIFAR-10/vgg16_kd.pth",
    "VGG-16_prun": "models_/CIFAR-10/vgg16_prun.pth",
    "VGG-16_quan": "models_/CIFAR-10/vgg16_quan.pth",
}

model_factories_mnist = {
    "SimpleCNN": SimpleCNN,
    "SimpleCNN_kd": SimpleCNN_Half,
    "SimpleCNN_prun": SimpleCNN,
    "SimpleCNN_quan": SimpleCNN,
    "LeNet-4": LeNet4,
    "LeNet-4_kd": LeNet4_Half,
    "LeNet-4_prun": LeNet4,
    "LeNet-4_quan": LeNet4,
    "LeNet-5": LeNet5,
    "LeNet-5_kd": LeNet5_Half,
    "LeNet-5_prun": LeNet5,
    "LeNet-5_quan": LeNet5,
}


def load_named_model(model_name, model_paths, model_factories):
    model = model_factories[model_name]()
    state_dict = torch.load(model_paths[model_name], map_location=device)
    model.load_state_dict(state_dict)
    return model


def model_pair_predict(org_model, dataloader, cps_model):
    org_model.to(device)
    cps_model.to(device)
    org_model.eval()
    cps_model.eval()
    diff = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            org_outputs = org_model(data)
            cps_outputs = cps_model(data)
            _, org_predicted = torch.max(org_outputs.data, 1)
            _, cps_predicted = torch.max(cps_outputs.data, 1)
            diff += (org_predicted != cps_predicted).sum().item()
    
    return diff
    
mnist_kd_pairs = [
    ("SimpleCNN", "SimpleCNN_kd"),
    ("LeNet-4", "LeNet-4_kd"),
    ("LeNet-5", "LeNet-5_kd"),
]


seed_indices = random.sample(range(len(mnist_test_loader.dataset)), 1000)
seed_mnist_subset = torch.utils.data.Subset(mnist_test_loader.dataset, seed_indices)
seed_mnist_test_loader = DataLoader(seed_mnist_subset, batch_size=1000, shuffle=False)

# Test
# Zero mutation.
for org_name, cps_name in mnist_kd_pairs:
    org_model = load_named_model(org_name, model_paths_mnist, model_factories_mnist)
    cps_model = load_named_model(cps_name, model_paths_mnist, model_factories_mnist)
    diff = model_pair_predict(org_model, seed_mnist_test_loader, cps_model)
    print(f"{org_name} vs {cps_name}: diff={diff}")

# 续写，attack mode 为a，不需要任何parser传参，就使用目前的step-by-step的写法
# 不要改动之前的代码，直接在下面续写攻击模式a的测试代码
# 参考test_gen_main.py中的攻击模式a的实现，直接在下面续写攻击模式a的测试代码

# What the fuck