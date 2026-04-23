import os

import numpy as np
import torch

from myLib.Inputs import SubsetInputs
from myLib.Result import PredictResult
from myUtils import create_folder, myLogger
from proj_utils import common_argparser
from test_gen_main import tf_gen
from model_loader.load_pytorch_models import load_org_and_cps_models


def create_predict_function_pytorch(org_model, cps_model, device):
    def predict(input_img):
        input_tensor = torch.from_numpy(input_img).float().to(device)
        with torch.no_grad():
            org_vec = org_model(input_tensor).detach().cpu().numpy()
            cps_vec = cps_model(input_tensor).detach().cpu().numpy()
        return PredictResult(org_vec), PredictResult(cps_vec)

    return predict


def preprocess_mnist_for_pytorch(img: np.ndarray) -> np.ndarray:
    working = img.astype(np.float32)
    if working.ndim == 2:
        working = working[np.newaxis, :, :]
    if working.ndim == 3:
        if working.shape[0] == 1:
            working = working
        else:
            working = np.transpose(working, (2, 0, 1))
        working = working[np.newaxis, ...]
    working = working / 255.0
    working = (working - 0.1307) / 0.3081
    return working


def preprocess_cifar_for_pytorch(img: np.ndarray) -> np.ndarray:
    working = img.astype(np.float32)
    if working.ndim == 3 and working.shape[0] == 3:
        working = np.transpose(working, (1, 2, 0))
    if working.ndim == 3:
        working = working[np.newaxis, ...]
    working = working / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    working = (working - mean) / std
    working = np.transpose(working, (0, 3, 1, 2))
    return working


def validate_dataset_arch(dataset, arch):
    valid_pairs = {
        ("mnist", "lenet1"),
        ("mnist", "lenet5"),
        ("cifar", "resnet"),
    }
    if (dataset, arch) not in valid_pairs:
        raise ValueError(f"Unsupported dataset/arch pair: dataset={dataset}, arch={arch}")


def main():
    parser = common_argparser()
    parser.add_argument("--cps_type", choices=["quan", "prun", "kd"], required=True)
    parser.add_argument("--backend", default="pytorch", choices=["pytorch"])
    args = parser.parse_args()

    validate_dataset_arch(args.dataset, args.arch)

    output_name = f"{args.arch}_{args.cps_type}_{args.seed}_{args.attack_mode}"
    save_dir = os.path.join(args.output_dir, f"{args.dataset}-pytorch", output_name)
    create_folder(save_dir)
    logger_file = os.path.join(args.output_dir, f"{args.dataset}-pytorch", f"{output_name}.log")
    logger = myLogger.create_logger(logger_file)
    logger(args)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger(f"Load models on device: {device}")
    org_model, cps_model = load_org_and_cps_models(args.dataset, args.arch, args.cps_type, device)

    predict_f = create_predict_function_pytorch(org_model, cps_model, device)
    input_sets = SubsetInputs(args.dataset, "tensorflow", args.arch, "quan-lite", args.seed)

    if args.dataset == "mnist":
        preprocessing = preprocess_mnist_for_pytorch
    elif args.dataset == "cifar":
        preprocessing = preprocess_cifar_for_pytorch
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    logger("Start the attack")
    tf_gen(args, input_sets, logger, save_dir, predict_f, preprocessing)


if __name__ == '__main__':
    main()
