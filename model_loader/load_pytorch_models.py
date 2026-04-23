import os
import torch

from model_definitions import SimpleCNN, LeNet4, LeNet5, PlainNet20, ResNet20, VGG16
from model_definitions import SimpleCNN_Half, LeNet4_Half, LeNet5_Half, PlainNet8, ResNet8, VGG11


def get_model_dir(dataset: str) -> str:
    if dataset == "mnist":
        return os.path.join("models_", "MNIST")
    if dataset == "cifar":
        return os.path.join("models_", "CIFAR-10")
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_model_builders(dataset: str):
    if dataset == "mnist":
        return {
            "lenet1": SimpleCNN,
            "simplecnn": SimpleCNN,
            "lenet4": LeNet4,
            "lenet5": LeNet5,
        }
    if dataset == "cifar":
        return {
            "resnet": ResNet20,
            "plainnet20": PlainNet20,
            "resnet20": ResNet20,
            "vgg16": VGG16,
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_compressed_builders(dataset: str):
    if dataset == "mnist":
        return {
            "kd": {
                "lenet1": SimpleCNN_Half,
                "simplecnn": SimpleCNN_Half,
                "lenet4": LeNet4_Half,
                "lenet5": LeNet5_Half,
            }
        }
    if dataset == "cifar":
        return {
            "kd": {
                "resnet": ResNet8,
                "plainnet20": PlainNet8,
                "resnet20": ResNet8,
                "vgg16": VGG11,
            }
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_original_checkpoint_name(dataset: str, arch: str) -> str:
    if dataset == "mnist":
        name_map = {
            "lenet1": "simplecnn.pth",
            "simplecnn": "simplecnn.pth",
            "lenet4": "lenet-4.pth",
            "lenet5": "lenet-5.pth",
        }
        return name_map[arch]
    if dataset == "cifar":
        name_map = {
            "resnet": "resnet20.pth",
            "plainnet20": "plainnet20.pth",
            "resnet20": "resnet20.pth",
            "vgg16": "vgg16.pth",
        }
        return name_map[arch]
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_compressed_checkpoint_name(dataset: str, arch: str, cps_type: str) -> str:
    if dataset == "mnist":
        suffix_map = {"quan": "_quan.pth", "prun": "_prun.pth", "kd": "_kd.pth"}
        base_map = {
            "lenet1": "simplecnn",
            "simplecnn": "simplecnn",
            "lenet4": "lenet-4",
            "lenet5": "lenet-5",
        }
        return f"{base_map[arch]}{suffix_map[cps_type]}"
    if dataset == "cifar":
        suffix_map = {"quan": "_quan.pth", "prun": "_prun.pth", "kd": "_kd.pth"}
        base_map = {
            "resnet": "resnet20",
            "plainnet20": "plainnet20",
            "resnet20": "resnet20",
            "vgg16": "vgg16",
        }
        return f"{base_map[arch]}{suffix_map[cps_type]}"
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_model_from_checkpoint(model, checkpoint_path: str, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_original_and_compressed_models(dataset: str, arch: str, cps_type: str, device: torch.device):
    arch = arch.lower()
    cps_type = cps_type.lower()

    model_dir = get_model_dir(dataset)
    original_builder = get_model_builders(dataset)[arch]
    compressed_builder = get_compressed_builders(dataset).get(cps_type, {}).get(arch, original_builder)

    original_model = original_builder()
    compressed_model = compressed_builder()

    original_path = os.path.join(model_dir, get_original_checkpoint_name(dataset, arch))
    compressed_path = os.path.join(model_dir, get_compressed_checkpoint_name(dataset, arch, cps_type))

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original checkpoint not found: {original_path}")
    if not os.path.exists(compressed_path):
        raise FileNotFoundError(f"Compressed checkpoint not found: {compressed_path}")

    original_model = load_model_from_checkpoint(original_model, original_path, device)
    compressed_model = load_model_from_checkpoint(compressed_model, compressed_path, device)
    return original_model, compressed_model, original_path, compressed_path
