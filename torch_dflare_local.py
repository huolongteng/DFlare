import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision import datasets

import model_definitions as model_zoo
from myLib.Result import PredictResult, SingleAttackResult
from myLib.fitnessValue import DiffProbFitnessValue, StateFitnessValue
from myLib.img_mutations import get_img_mutations
from myLib.probability_img_mutations import ProbabilityImgMutations, RandomImgMutations
from myUtils import create_folder, myLogger
from proj_utils import summary_attack_results


MODEL_PATHS = {
    "mnist": {
        "lenet-4": {
            "org": "./models_/MNIST/lenet-4.pth",
            "kd": "./models_/MNIST/lenet-4_kd.pth",
            "quant": "./models_/MNIST/lenet-4_quant.pth",
            "prune": "./models_/MNIST/lenet-4_prune.pt",
        },
        "lenet-5": {
            "org": "./models_/MNIST/lenet-5.pth",
            "kd": "./models_/MNIST/lenet-5_kd.pth",
            "quant": "./models_/MNIST/lenet-5_quant.pth",
            "prune": "./models_/MNIST/lenet-5_prune.pt",
        },
        "simplecnn": {
            "org": "./models_/MNIST/simplecnn.pth",
            "kd": "./models_/MNIST/simplecnn_kd.pth",
            "quant": "./models_/MNIST/simplecnn_quant.pth",
            "prune": "./models_/MNIST/simplecnn_prune.pt",
        },
    },
    "cifar": {
        "resnet20": {
            "org": "./models_/CIFAR-10/resnet20.pth",
            "kd": "./models_/CIFAR-10/resnet20_kd.pt",
            "quant": "./models_/CIFAR-10/resnet20_quant.pth",
            "prune": "./models_/CIFAR-10/resnet20_prune.pt",
        },
        "plainnet20": {
            "org": "./models_/CIFAR-10/plainnet20.pth",
            "kd": "./models_/CIFAR-10/plainnet20_kd.pt",
            "quant": "./models_/CIFAR-10/plainnet20_quant.pth",
            "prune": "./models_/CIFAR-10/plainnet20_prune.pt",
        },
        "vgg16": {
            "org": "./models_/CIFAR-10/vgg16.pth",
            "kd": "./models_/CIFAR-10/vgg16_kd.pt",
            "quant": "./models_/CIFAR-10/vgg16_quant.pth",
            "prune": "./models_/CIFAR-10/vgg16_prune.pt",
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Local PyTorch DFlare runner for original + compressed models")
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--arch", required=True,
                        help="mnist: lenet-4, lenet-5, simplecnn; cifar: resnet20, plainnet20, vgg16")
    parser.add_argument("--cps-type", choices=["kd", "quant", "prune"], default="kd")
    parser.add_argument("--maxit", type=int, default=100)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results_torch")
    parser.add_argument("--attack-mode", type=str, default="a", choices=["a", "b", "c", "d"])
    parser.add_argument("--timeout", type=int, default=240, help="timeout for each seed input in seconds")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--coverage-threshold", type=float, default=0.50)
    return parser.parse_args()


def canonical_arch(dataset: str, arch: str) -> str:
    aliases = {
        "mnist": {
            "lenet4": "lenet-4",
            "lenet-4": "lenet-4",
            "lenet5": "lenet-5",
            "lenet-5": "lenet-5",
            "simplecnn": "simplecnn",
            "cnn": "simplecnn",
        },
        "cifar": {
            "resnet20": "resnet20",
            "plain20": "plainnet20",
            "plainnet20": "plainnet20",
            "vgg16": "vgg16",
        },
    }
    try:
        return aliases[dataset][arch.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported arch '{arch}' for dataset '{dataset}'") from exc


class TorchvisionInputs:
    def __init__(self, dataset: str, data_dir: str, num: int, random_seed: int):
        if dataset == "mnist":
            self.dataset = datasets.MNIST(root=data_dir, train=False, download=True)
        elif dataset == "cifar":
            self.dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
        else:
            raise ValueError(f"Unsupported dataset {dataset}")
        self.indices = self._build_balanced_indices(num, random_seed)

    def _build_balanced_indices(self, num: int, random_seed: int):
        total_len = len(self.dataset)
        if num >= total_len:
            return list(range(total_len))

        targets = self.dataset.targets
        if torch.is_tensor(targets):
            labels = targets.cpu().numpy().astype(int)
        else:
            labels = np.asarray(targets, dtype=int)

        classes = np.unique(labels)
        rng = np.random.RandomState(random_seed)
        class_order = classes.copy()
        rng.shuffle(class_order)

        per_class = num // len(classes)
        remainder = num % len(classes)
        quotas = {cls: per_class for cls in classes}
        for cls in class_order[:remainder]:
            quotas[int(cls)] += 1

        selected = []
        for cls in class_order:
            cls_indices = np.where(labels == cls)[0]
            rng.shuffle(cls_indices)
            take = quotas[int(cls)]
            selected.extend(cls_indices[:take].tolist())

        rng.shuffle(selected)
        return selected

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        img = np.array(img)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        return {"img": img.astype(np.uint8), "label": int(label)}

    @property
    def len(self):
        return len(self.indices)


class ExactCoveredStates:
    def __init__(self, threshold=0.50):
        self.threshold = threshold
        self.corpus = []

    def update_function(self, element):
        element = np.asarray(element, dtype=np.float32).ravel()
        if len(self.corpus) == 0:
            self.corpus.append(element)
            return True, 100.0

        lookup = np.vstack(self.corpus)
        distances = np.sum(np.square(lookup - element), axis=1)
        nearest_distance = float(np.min(distances))
        if nearest_distance > self.threshold:
            self.corpus.append(element)
            return True, nearest_distance
        return False, nearest_distance


class TwoConvClassifier(nn.Module):
    def __init__(self, conv1_out: int, conv2_out: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.fc1 = nn.Linear(conv2_out * 4 * 4, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        return self.fc1(x)


class LeNet5Variant(nn.Module):
    def __init__(self, conv1_out: int, conv2_out: int, fc1_out: int, fc2_out: int, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(conv2_out * 5 * 5, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class VGGVariant(nn.Module):
    CFGS = {
        "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
        "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    }

    def __init__(self, variant: str, num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers(self.CFGS[variant])
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for value in cfg:
            if value == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, value, kernel_size=3, padding=1),
                    nn.BatchNorm2d(value),
                    nn.ReLU(inplace=True),
                ])
                in_channels = value
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def infer_resnet_blocks(state_dict):
    blocks = []
    for layer_name in ("layer1", "layer2", "layer3"):
        indices = []
        prefix = f"{layer_name}."
        for key in state_dict.keys():
            if key.startswith(prefix):
                parts = key.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    indices.append(int(parts[1]))
        blocks.append(max(indices) + 1 if indices else 0)
    return blocks


def build_original_model(dataset: str, arch: str):
    builders = {
        "mnist": {
            "lenet-4": model_zoo.LeNet4,
            "lenet-5": model_zoo.LeNet5,
            "simplecnn": model_zoo.SimpleCNN,
        },
        "cifar": {
            "resnet20": model_zoo.ResNet20,
            "plainnet20": model_zoo.PlainNet20,
            "vgg16": model_zoo.VGG16,
        },
    }
    return builders[dataset][arch]()


def build_example_inputs(dataset: str):
    if dataset == "mnist":
        return (torch.zeros(1, 1, 28, 28),)
    return (torch.zeros(1, 3, 32, 32),)


def build_quantized_model(dataset: str, arch: str, org_state):
    model = build_original_model(dataset, arch)
    model.load_state_dict(org_state)
    model.cpu().eval()

    qconfig_mapping = get_default_qconfig_mapping("onednn")
    prepared = prepare_fx(model, qconfig_mapping, build_example_inputs(dataset))
    prepared(*build_example_inputs(dataset))
    quantized = convert_fx(prepared)
    return quantized


def infer_kd_model(dataset: str, state_dict):
    if dataset == "mnist":
        if "fc3.weight" in state_dict:
            return LeNet5Variant(
                conv1_out=state_dict["conv1.weight"].shape[0],
                conv2_out=state_dict["conv2.weight"].shape[0],
                fc1_out=state_dict["fc1.weight"].shape[0],
                fc2_out=state_dict["fc2.weight"].shape[0],
                num_classes=state_dict["fc3.weight"].shape[0],
            )
        return TwoConvClassifier(
            conv1_out=state_dict["conv1.weight"].shape[0],
            conv2_out=state_dict["conv2.weight"].shape[0],
            num_classes=state_dict["fc1.weight"].shape[0],
        )

    if dataset == "cifar":
        if "linear.weight" in state_dict:
            blocks = infer_resnet_blocks(state_dict)
            base_channels = state_dict["conv1.weight"].shape[0]
            has_shortcut = any(".shortcut." in key for key in state_dict)
            if base_channels < 16 and blocks == [1, 1, 1]:
                if has_shortcut:
                    return model_zoo.ResNetStudent(model_zoo.BasicBlock, blocks, base_channels=base_channels,
                                                   num_classes=state_dict["linear.weight"].shape[0])
                return model_zoo.PlainNetStudent(model_zoo.PlainBlock, blocks, base_channels=base_channels,
                                                 num_classes=state_dict["linear.weight"].shape[0])
            return model_zoo.ResNet(model_zoo.BasicBlock, blocks, num_classes=state_dict["linear.weight"].shape[0])
        conv_keys = [k for k, v in state_dict.items() if k.startswith("features.") and k.endswith(".weight") and v.ndim == 4]
        if len(conv_keys) == 5:
            channels = [state_dict[key].shape[0] for key in conv_keys]
            return model_zoo.VGGStudent(channels, num_classes=state_dict["classifier.weight"].shape[0])
        if len(conv_keys) == 13:
            return VGGVariant("VGG16", num_classes=state_dict["classifier.weight"].shape[0])
        if len(conv_keys) == 8:
            return VGGVariant("VGG11", num_classes=state_dict["classifier.weight"].shape[0])
    raise ValueError("Unable to infer KD model architecture from checkpoint")


def load_models(dataset: str, arch: str, cps_type: str, device: torch.device):
    paths = MODEL_PATHS[dataset][arch]
    org_state = torch.load(paths["org"], map_location="cpu")

    org_model = build_original_model(dataset, arch)
    org_model.load_state_dict(org_state)
    org_model.to(device).eval()

    if cps_type == "kd":
        if paths["kd"].endswith(".pt"):
            cps_model = torch.load(paths["kd"], map_location="cpu", weights_only=False)
        else:
            cps_state = torch.load(paths["kd"], map_location="cpu")
            cps_model = infer_kd_model(dataset, cps_state)
            cps_model.load_state_dict(cps_state)
        cps_device = device
    elif cps_type == "quant":
        cps_state = torch.load(paths["quant"], map_location="cpu")
        cps_model = build_quantized_model(dataset, arch, org_state)
        cps_model.load_state_dict(cps_state)
        cps_device = torch.device("cpu")
    elif cps_type == "prune":
        cps_model = torch.load(paths["prune"], map_location="cpu", weights_only=False)
        cps_device = device
    else:
        raise ValueError(f"Unsupported compressed model type '{cps_type}'")

    cps_model.to(cps_device).eval()
    return org_model, cps_model, device, cps_device


def preprocess_mnist(img: np.ndarray, device: torch.device):
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def preprocess_cifar(img: np.ndarray, device: torch.device):
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=tensor.dtype).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device)


def create_predict_function(dataset: str, org_model: nn.Module, cps_model: nn.Module,
                            org_device: torch.device, cps_device: torch.device):
    preprocess = preprocess_mnist if dataset == "mnist" else preprocess_cifar

    def predict(input_img):
        org_tensor = preprocess(input_img, org_device)
        cps_tensor = org_tensor if cps_device == org_device else preprocess(input_img, cps_device)
        with torch.no_grad():
            org_vec = org_model(org_tensor).detach().cpu().numpy()
            cps_vec = cps_model(cps_tensor).detach().cpu().numpy()
        return PredictResult(org_vec), PredictResult(cps_vec)

    return predict


def run_attack(args, inputs_set, logger, save_dir, predict_f):
    overall_start_time = time.time()
    number_of_data = min(inputs_set.len, args.num)
    success_iter = np.ones([number_of_data]) * -1
    seed_times = np.zeros([number_of_data])

    for idx in range(number_of_data):
        seed_file = inputs_set[idx]
        raw_seed_input = seed_file["img"]
        seed_label = seed_file["label"]

        logger("Img idx {} label {}".format(idx, seed_label))

        if args.attack_mode in ("a", "b", "d"):
            covered_states = ExactCoveredStates(threshold=args.coverage_threshold)
        elif args.attack_mode != "c":
            raise NotImplementedError

        mutation = get_img_mutations()
        if args.attack_mode in ("a", "c", "d"):
            p_mutation = ProbabilityImgMutations(mutation, args.seed)
        elif args.attack_mode == "b":
            p_mutation = RandomImgMutations(mutation, args.seed)
        else:
            raise NotImplementedError

        seed_org_result, seed_cps_result = predict_f(raw_seed_input)
        result = SingleAttackResult(raw_seed_input, seed_label, idx, seed_org_result, seed_cps_result, save_dir)
        start_time = time.time()

        if seed_org_result.label != seed_cps_result.label:
            logger("No need to search: org: {} vs cps: {}".format(seed_org_result.label, seed_cps_result.label))
            success_iter[idx] = 0
            result.update_results(None, seed_org_result, seed_cps_result, 0)
        else:
            if args.attack_mode in ("a", "b", "d"):
                covered_states.update_function(np.hstack([seed_org_result.vec, seed_cps_result.vec]))
                best_fitness_value = StateFitnessValue(False, 0)
            else:
                best_fitness_value = DiffProbFitnessValue(-1)

            latest_img = np.copy(raw_seed_input)
            last_mutation_operator = None

            for iteration in range(1, args.maxit + 1):
                logger("Iteration {}".format(iteration))
                if time.time() - start_time > args.timeout:
                    logger("Time Out")
                    break

                m = p_mutation.choose_mutator(last_mutation_operator)
                m.total += 1
                logger("Mutator :{}".format(m.name))
                new_img = m.mut(np.copy(latest_img))

                org_result, cps_result = predict_f(new_img)
                if org_result.label != cps_result.label:
                    logger("Found: org: {} vs cps: {}".format(org_result.label, cps_result.label))
                    success_iter[idx] = iteration
                    m.delta_bigger_than_zero += 1
                    result.update_results(new_img, org_result, cps_result, iteration)
                    break

                diff_prob = org_result.prob[0] - cps_result.prob[0]
                if args.attack_mode in ("a", "b", "d"):
                    coverage = np.hstack([org_result.vec, cps_result.vec])
                    add_to_corpus, distance = covered_states.update_function(coverage)
                    fitness_value = StateFitnessValue(add_to_corpus, diff_prob)
                else:
                    fitness_value = DiffProbFitnessValue(diff_prob)

                if fitness_value.better_than(best_fitness_value):
                    best_fitness_value = fitness_value
                    m.delta_bigger_than_zero += 1
                    latest_img = np.copy(new_img)
                    last_mutation_operator = m.name
                    logger("update fitness value to {}".format(best_fitness_value))

                logger("Best " + str(best_fitness_value))

        result.save()
        seed_times[idx] = time.time() - start_time

    total_runtime = time.time() - overall_start_time
    summary_attack_results(success_iter, logger, args.attack_mode, seed_times=seed_times, total_runtime=total_runtime)


def main():
    args = parse_args()
    args.arch = canonical_arch(args.dataset, args.arch)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_name = "{}_{}_{}_{}".format(args.arch, args.cps_type, args.seed, args.attack_mode)
    save_dir = os.path.join(args.output_dir, f"{args.dataset}-torch", output_name)
    create_folder(save_dir)

    log_path = os.path.join(args.output_dir, f"{args.dataset}-torch", f"{output_name}.log")
    logger = myLogger.create_logger(log_path)
    logger(args)
    logger("Load models")

    org_model, cps_model, org_device, cps_device = load_models(args.dataset, args.arch, args.cps_type, device)
    logger(f"Using org device {org_device}; compressed model device {cps_device}")
    predict_f = create_predict_function(args.dataset, org_model, cps_model, org_device, cps_device)

    logger("Load dataset")
    inputs_set = TorchvisionInputs(args.dataset, args.data_dir, num=args.num, random_seed=args.seed)

    logger("Start the attack")
    print("Start the attack")
    run_attack(args, inputs_set, logger, save_dir, predict_f)


if __name__ == "__main__":
    main()
