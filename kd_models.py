import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import models (Teachers and Students)
from model_definitions import (
    SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16,
    SimpleCNN_Half, LeNet4_Half, LeNet5_Half, PlainNet8, ResNet8, VGG11
)

# === General Settings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

mnist_dir = os.path.join('models_', 'MNIST')
cifar_dir = os.path.join('models_', 'CIFAR-10')

final_kd_accuracies = {}


def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total if total > 0 else 0.0

# === Knowledge Distillation Loss Function ===
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    """
    Computes Knowledge Distillation Loss.
    T: Temperature to soften probabilities.
    alpha: Weight for the original Cross-Entropy loss.
    """
    # Standard Cross-Entropy Loss
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # KL Divergence Loss
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_softmax = F.log_softmax(student_logits / T, dim=1)
    kl_loss = F.kl_div(student_log_softmax, soft_targets, reduction='batchmean') * (T ** 2)
    
    return alpha * ce_loss + (1.0 - alpha) * kl_loss

# === Load Models configuration (Teacher_Name : (Teacher_Model, Student_Model, Dir)) ===
kd_pairs_mnist = {
    "SimpleCNN": (SimpleCNN(), SimpleCNN_Half(), mnist_dir),
    "LeNet-4": (LeNet4(), LeNet4_Half(), mnist_dir),
    "LeNet-5": (LeNet5(), LeNet5_Half(), mnist_dir)
}

kd_pairs_cifar = {
    "PlainNet-20": (PlainNet20(), PlainNet8(), cifar_dir),
    "ResNet-20": (ResNet20(), ResNet8(), cifar_dir),
    "VGG-16": (VGG16(), VGG11(), cifar_dir)
}

# ====================================================================
# MNIST Knowledge Distillation
# ====================================================================
print("\n--- Processing MNIST Knowledge Distillation ---")
transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=1000, shuffle=False)

for t_name, (teacher, student, d) in kd_pairs_mnist.items():
    teacher_path = os.path.join(d, f'{t_name.lower()}.pth')
    if not os.path.exists(teacher_path):
        print(f"Teacher model {t_name} not found at {teacher_path}. Skipping.")
        continue
        
    print(f"\nDistilling {t_name} -> {t_name}_kd ...")
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device)
    teacher.eval() # Teacher should always be in eval mode
    
    student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    num_epochs = 3
    
    # Training Loop
    student.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(mnist_train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(data)
            student_logits = student(data)
            
            loss = distillation_loss(student_logits, teacher_logits, target, T=4.0, alpha=0.5)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(mnist_train_loader)}], Loss: {loss.item():.4f}")

    # Evaluating Student
    accuracy = evaluate_model(student, mnist_test_loader, device)
    kd_name = f"{t_name}_kd"
    final_kd_accuracies[kd_name] = accuracy
    print(f"Student Accuracy of {kd_name} on test set: {accuracy:.2f}%")

    student_path = os.path.join(d, f'{kd_name.lower()}.pth')
    torch.save(student.state_dict(), student_path)
    print(f"Saved student model to {student_path}")

# ====================================================================
# CIFAR-10 Knowledge Distillation
# ====================================================================
print("\n--- Processing CIFAR-10 Knowledge Distillation ---")
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
cifar_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar_train)
cifar_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar_test)
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=128, shuffle=True)
cifar_test_loader = DataLoader(cifar_test_dataset, batch_size=100, shuffle=False)

for t_name, (teacher, student, d) in kd_pairs_cifar.items():
    teacher_path = os.path.join(d, f'{t_name.lower().replace("-", "")}.pth')
    if not os.path.exists(teacher_path):
        print(f"Teacher model {t_name} not found at {teacher_path}. Skipping.")
        continue
        
    print(f"\nDistilling {t_name} -> {t_name}_kd ...")
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.to(device)
    teacher.eval() # Frozen
    
    # 针对较小的学生模型做超参设置
    student.to(device)
    # 对于 CIFAR-10 我们同样使用较多 epoch 进行训练，比如对于 ResNet8 和 PlainNet8 使用学习率0.1训练。
    if t_name in ["PlainNet-20", "ResNet-20"]:
        lr = 0.1
        num_epochs = 10
    else:
        lr = 0.01
        num_epochs = 10
        
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    student.train()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(cifar_train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)
            
            loss = distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5)
            loss.backward()
            optimizer.step()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(cifar_train_loader)}], Loss: {loss.item():.4f}")
        scheduler.step()

    accuracy = evaluate_model(student, cifar_test_loader, device)
    kd_name = f"{t_name}_kd"
    final_kd_accuracies[kd_name] = accuracy
    print(f"Student Accuracy of {kd_name} on the test set: {accuracy:.2f}%")

    # Save student model
    student_path = os.path.join(d, f'{kd_name.lower().replace("-", "")}.pth')
    torch.save(student.state_dict(), student_path)
    print(f"Saved student model to {student_path}")


print("\n" + "="*50)
print("FINAL KNOWLEDGE DISTILLATION (STUDENT) ACCURACIES")
print("="*50)
for kd_name, acc in final_kd_accuracies.items():
    print(f"{kd_name:<15}: {acc:.2f}%")
print("="*50)
print("Knowledge Distillation completed.")
