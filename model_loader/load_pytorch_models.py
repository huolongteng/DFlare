import os
from typing import Tuple

import torch

from model_definitions import SimpleCNN, LeNet5, ResNet20, SimpleCNN_Half, LeNet5_Half, ResNet8


def _build_original_model(dataset: str, arch: str):
    if dataset == "mnist" and arch == "lenet1":
        return SimpleCNN()
    if dataset == "mnist" and arch == "lenet5":
        return LeNet5()
    if dataset == "cifar" and arch == "resnet":
        return ResNet20()
    raise ValueError(f"Unsupported original model pair: dataset={dataset}, arch={arch}")


def _build_compressed_model(dataset: str, arch: str, cps_type: str):
    if cps_type == "kd":
        if dataset == "mnist" and arch == "lenet1":
            return SimpleCNN_Half()
        if dataset == "mnist" and arch == "lenet5":
            return LeNet5_Half()
        if dataset == "cifar" and arch == "resnet":
            return ResNet8()
    return _build_original_model(dataset, arch)


def _resolve_weight_paths(dataset: str, arch: str, cps_type: str) -> Tuple[str, str]:
    if dataset == "mnist" and arch == "lenet1":
        org_path = os.path.join("models_", "MNIST", "simplecnn.pth")
        cps_path = {
            "quan": os.path.join("models_", "MNIST", "simplecnn_quan.pth"),
            "prun": os.path.join("models_", "MNIST", "simplecnn_prun.pth"),
            "kd": os.path.join("models_", "MNIST", "simplecnn_kd.pth"),
        }[cps_type]
        return org_path, cps_path

    if dataset == "mnist" and arch == "lenet5":
        org_path = os.path.join("models_", "MNIST", "lenet5.pth")
        cps_path = {
            "quan": os.path.join("models_", "MNIST", "lenet5_quan.pth"),
            "prun": os.path.join("models_", "MNIST", "lenet5_prun.pth"),
            "kd": os.path.join("models_", "MNIST", "lenet-5_kd.pth"),
        }[cps_type]
        return org_path, cps_path

    if dataset == "cifar" and arch == "resnet":
        org_path = os.path.join("models_", "CIFAR-10", "resnet20.pth")
        cps_path = {
            "quan": os.path.join("models_", "CIFAR-10", "resnet20_quan.pth"),
            "prun": os.path.join("models_", "CIFAR-10", "resnet20_prun.pth"),
            "kd": os.path.join("models_", "CIFAR-10", "resnet20_kd.pth"),
        }[cps_type]
        return org_path, cps_path

    raise ValueError(f"Unsupported pair: dataset={dataset}, arch={arch}, cps_type={cps_type}")


def _load_state_dict_from_path(model: torch.nn.Module, path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Weight file does not exist: {path}. "
            f"Please run train_models.py and corresponding compression script first."
        )
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)


def load_org_and_cps_models(dataset: str, arch: str, cps_type: str, device: torch.device):
    org_model = _build_original_model(dataset, arch)
    cps_model = _build_compressed_model(dataset, arch, cps_type)

    org_path, cps_path = _resolve_weight_paths(dataset, arch, cps_type)
    _load_state_dict_from_path(org_model, org_path, device)
    _load_state_dict_from_path(cps_model, cps_path, device)

    org_model = org_model.to(device).eval()
    cps_model = cps_model.to(device).eval()
    return org_model, cps_model
