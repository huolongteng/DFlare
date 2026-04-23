import os
from typing import Tuple

import numpy as np
import torch

from model_definitions import (
    LeNet5,
    LeNet5_Half,
    ResNet20,
    ResNet8,
    SimpleCNN,
    SimpleCNN_Half,
)


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]


def _build_model(dataset: str, arch: str, cps_type: str):
    if dataset == "mnist":
        if arch == "lenet1":
            if cps_type == "kd":
                return SimpleCNN_Half()
            return SimpleCNN()
        if arch == "lenet5":
            if cps_type == "kd":
                return LeNet5_Half()
            return LeNet5()
    if dataset == "cifar" and arch == "resnet":
        if cps_type == "kd":
            return ResNet8()
        return ResNet20()
    raise ValueError(f"Unsupported dataset/arch/cps_type: {dataset}/{arch}/{cps_type}")


def _org_model_filename(dataset: str, arch: str) -> str:
    if dataset == "mnist" and arch == "lenet1":
        return "simplecnn.pth"
    if dataset == "mnist" and arch == "lenet5":
        return "lenet-5.pth"
    if dataset == "cifar" and arch == "resnet":
        return "resnet20.pth"
    raise ValueError(f"Unsupported dataset/arch: {dataset}/{arch}")


def _cps_model_filename(dataset: str, arch: str, cps_type: str) -> str:
    if cps_type not in {"quan", "prun", "kd"}:
        raise ValueError(f"Unsupported compression type: {cps_type}")

    suffix = f"_{cps_type}.pth"

    if dataset == "mnist" and arch == "lenet1":
        return f"simplecnn{suffix}"
    if dataset == "mnist" and arch == "lenet5":
        return f"lenet-5{suffix}"
    if dataset == "cifar" and arch == "resnet":
        return f"resnet20{suffix}"
    raise ValueError(f"Unsupported dataset/arch: {dataset}/{arch}")


def _dataset_model_dir(dataset: str, model_root: str) -> str:
    if dataset == "mnist":
        return os.path.join(model_root, "MNIST")
    if dataset == "cifar":
        return os.path.join(model_root, "CIFAR-10")
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_model_pair(dataset: str, arch: str, cps_type: str, model_root: str, device: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model_dir = _dataset_model_dir(dataset, model_root)
    org_path = os.path.join(model_dir, _org_model_filename(dataset, arch))
    cps_path = os.path.join(model_dir, _cps_model_filename(dataset, arch, cps_type))

    if not os.path.exists(org_path):
        raise FileNotFoundError(f"Original model file not found: {org_path}")
    if not os.path.exists(cps_path):
        raise FileNotFoundError(f"Compressed model file not found: {cps_path}")

    org_model = _build_model(dataset, arch, "org")
    cps_model = _build_model(dataset, arch, cps_type)

    org_model.load_state_dict(torch.load(org_path, map_location=device))
    cps_model.load_state_dict(torch.load(cps_path, map_location=device))

    org_model.to(device).eval()
    cps_model.to(device).eval()
    return org_model, cps_model


def preprocess_seed_input(dataset: str, image: np.ndarray) -> torch.Tensor:
    np_img = image.astype(np.float32)

    if dataset == "mnist":
        if np_img.ndim == 2:
            np_img = np_img[np.newaxis, ...]
        if np_img.ndim == 3 and np_img.shape[0] != 1:
            np_img = np.transpose(np_img, (2, 0, 1))
        if np_img.max() > 1.0:
            np_img = np_img / 255.0
        np_img = (np_img - MNIST_MEAN) / MNIST_STD
        np_img = np_img[np.newaxis, ...]
    elif dataset == "cifar":
        if np_img.ndim == 3 and np_img.shape[0] in [1, 3]:
            np_img = np.transpose(np_img, (1, 2, 0))
        if np_img.max() > 1.0:
            np_img = np_img / 255.0
        for channel in range(3):
            np_img[..., channel] = (np_img[..., channel] - CIFAR_MEAN[channel]) / CIFAR_STD[channel]
        np_img = np.transpose(np_img, (2, 0, 1))
        np_img = np_img[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return torch.from_numpy(np_img).float()
