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


# def model_pair_predict(org_model, dataloader, cps_model):
#     org_model.to(device)
#     cps_model.to(device)
#     org_model.eval()
#     cps_model.eval()
#     diff = 0
    
#     with torch.no_grad():
#         for data, target in dataloader:
#             data, target = data.to(device), target.to(device)
#             org_outputs = org_model(data)
#             cps_outputs = cps_model(data)
#             _, org_predicted = torch.max(org_outputs.data, 1)
#             _, cps_predicted = torch.max(cps_outputs.data, 1)
#             diff += (org_predicted != cps_predicted).sum().item()
    
#     return diff
    
mnist_kd_pairs = [
    ("SimpleCNN", "SimpleCNN_kd"),
    ("LeNet-4", "LeNet-4_kd"),
    ("LeNet-5", "LeNet-5_kd"),
]


seed_indices = random.sample(range(len(mnist_test_loader.dataset)), 1000)
seed_mnist_subset = torch.utils.data.Subset(mnist_test_loader.dataset, seed_indices)
seed_mnist_test_loader = DataLoader(seed_mnist_subset, batch_size=1000, shuffle=False)




# Test
from myLib.img_mutations import get_img_mutations
from myLib.probability_img_mutations import ProbabilityImgMutations
from myLib.fitnessValue import StateFitnessValue
from myLib.Result import PredictResult
import time


class CoveredStatesFallback:
    """
    A lightweight replacement for CoveredStates used when pyflann is unavailable.
    The interface matches update_function(element) -> (add_to_corpus, distance).
    """

    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.corpus = []

    def update_function(self, element):
        element = np.asarray(element).reshape(-1)
        if len(self.corpus) == 0:
            self.corpus.append(element)
            return True, 100

        distances = [np.sum(np.square(element - c)) for c in self.corpus]
        nearest_distance = min(distances)
        if nearest_distance > self.threshold:
            self.corpus.append(element)
            return True, nearest_distance
        return False, nearest_distance


def tensor_to_uint8_img(tensor_img):
    # tensor_img: [1, 28, 28], normalized by (x - 0.1307) / 0.3081
    img = tensor_img.squeeze(0).cpu().numpy()
    img = (img * 0.3081 + 0.1307) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img[..., np.newaxis]


def uint8_img_to_tensor(img):
    # img: [28, 28, 1], uint8
    gray = img.squeeze(-1).astype(np.float32) / 255.0
    gray = (gray - 0.1307) / 0.3081
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)


def predict_pair(org_model, cps_model, img_uint8):
    input_tensor = uint8_img_to_tensor(img_uint8)
    with torch.no_grad():
        org_vec = org_model(input_tensor).detach().cpu().numpy()
        cps_vec = cps_model(input_tensor).detach().cpu().numpy()
    return PredictResult(org_vec), PredictResult(cps_vec)


print("\n===== Attack Mode a Test (step-by-step, no parser args) =====")
attack_mode_a_maxit = 80
attack_mode_a_timeout = 10.0  # seconds per seed, aligned with tf_gen timeout control
attack_mode_a_runs = 10


def scalar_prob(prob_value):
    """Convert PredictResult.prob (scalar/array) to float for fitness computation."""
    arr = np.asarray(prob_value).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0


for org_name, cps_name in mnist_kd_pairs:
    print(f"\n[Attack Mode a] {org_name} vs {cps_name}")

    org_model = load_named_model(org_name, model_paths_mnist, model_factories_mnist).to(device)
    cps_model = load_named_model(cps_name, model_paths_mnist, model_factories_mnist).to(device)
    org_model.eval()
    cps_model.eval()

    run_success_rates = []
    total_success_time = 0.0
    total_success_query = 0
    total_success_attacks = 0

    for run_idx in range(attack_mode_a_runs):
        run_seed = 42 + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

        success_count = 0
        already_diff_count = 0
        total_count = 0
        attack_success_count = 0
        attack_trial_count = 0

        for data, _ in seed_mnist_test_loader:
            batch_size = data.size(0)
            for i in range(batch_size):
                raw_seed_tensor = data[i]
                raw_seed_img = tensor_to_uint8_img(raw_seed_tensor)

                covered_states = CoveredStatesFallback()
                mutation = get_img_mutations()
                p_mutation = ProbabilityImgMutations(mutation, random_seed=run_seed + total_count)

                seed_org_result, seed_cps_result = predict_pair(org_model, cps_model, raw_seed_img)

                if seed_org_result.label != seed_cps_result.label:
                    already_diff_count += 1
                    success_count += 1
                    total_count += 1
                    continue

                attack_trial_count += 1
                start_time = time.time()
                _, _ = covered_states.update_function(np.hstack([seed_org_result.vec, seed_cps_result.vec]))
                best_fitness_value = StateFitnessValue(False, 0)
                latest_img = np.copy(raw_seed_img)
                last_mutation_operator = None

                found = False
                success_query = 0
                for iteration in range(1, attack_mode_a_maxit + 1):
                    if time.time() - start_time > attack_mode_a_timeout:
                        break

                    m = p_mutation.choose_mutator(last_mutation_operator)
                    m.total += 1

                    new_img = m.mut(np.copy(latest_img))
                    org_result, cps_result = predict_pair(org_model, cps_model, new_img)

                    if org_result.label != cps_result.label:
                        m.delta_bigger_than_zero += 1
                        found = True
                        success_query = iteration
                        break

                    diff_prob = scalar_prob(org_result.prob) - scalar_prob(cps_result.prob)
                    coverage = np.hstack([org_result.vec, cps_result.vec])
                    add_to_corpus, _ = covered_states.update_function(coverage)
                    fitness_value = StateFitnessValue(add_to_corpus, diff_prob)

                    if fitness_value.better_than(best_fitness_value):
                        best_fitness_value = fitness_value
                        m.delta_bigger_than_zero += 1
                        latest_img = np.copy(new_img)
                        last_mutation_operator = m.name

                if found:
                    success_count += 1
                    attack_success_count += 1
                    total_success_attacks += 1
                    total_success_time += (time.time() - start_time)
                    total_success_query += success_query

                total_count += 1

        run_success_rate = attack_success_count / attack_trial_count if attack_trial_count > 0 else 0.0
        run_success_rates.append(run_success_rate)
        print(
            f"run={run_idx + 1}/{attack_mode_a_runs}, seed={run_seed}, "
            f"seeds={total_count}, already_diff={already_diff_count}, "
            f"attack_trials={attack_trial_count}, attack_success={attack_success_count}, "
            f"success_rate={run_success_rate:.4f}"
        )

    average_success_rate = float(np.mean(run_success_rates)) if len(run_success_rates) > 0 else 0.0
    average_time = total_success_time / total_success_attacks if total_success_attacks > 0 else 0.0
    average_query = total_success_query / total_success_attacks if total_success_attacks > 0 else 0.0
    print(
        f"[Attack Mode a Average over {attack_mode_a_runs} runs] "
        f"avg_success_rate={average_success_rate:.4f}, "
        f"avg_time={average_time:.6f}s, avg_query={average_query:.2f}, "
        f"total_success_attacks={total_success_attacks}"
    )
