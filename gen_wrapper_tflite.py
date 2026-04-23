import os
import numpy as np
import torch

from myLib.Inputs import SubsetInputs
from myLib.Result import PredictResult
from myUtils import create_folder, myLogger
from proj_utils import common_argparser
from test_gen_main import tf_gen
from model_loader.load_pytorch_models import load_original_and_compressed_models


def convert_numpy_to_nchw(img: np.ndarray, dataset: str) -> np.ndarray:
    # Step 1: convert input to float32.
    img = img.astype(np.float32)

    # Step 2: add a batch dimension for a single image.
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
    if img.ndim == 3:
        if dataset == "mnist":
            # MNIST may be HWC(28,28,1), CHW(1,28,28), or HW(28,28).
            if img.shape[-1] == 1:
                img = np.transpose(img, (2, 0, 1))
        else:
            # CIFAR may be HWC(32,32,3) or CHW(3,32,32).
            if img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]

    # Step 3: ensure channel-first batch format.
    if img.ndim != 4:
        raise ValueError(f"Unexpected image shape after preprocessing: {img.shape}")

    return img


def normalize_for_dataset(img_nchw: np.ndarray, dataset: str) -> np.ndarray:
    # Step 4: convert uint8-like range to [0, 1].
    if np.max(img_nchw) > 1.0:
        img_nchw = img_nchw / 255.0

    # Step 5: normalize with the same stats used in model training/evaluation.
    if dataset == "mnist":
        mean = np.array([0.1307], dtype=np.float32).reshape(1, 1, 1, 1)
        std = np.array([0.3081], dtype=np.float32).reshape(1, 1, 1, 1)
    elif dataset == "cifar":
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return (img_nchw - mean) / std


def build_preprocessing(dataset: str):
    def preprocessing(img: np.ndarray) -> np.ndarray:
        img_nchw = convert_numpy_to_nchw(img, dataset)
        return normalize_for_dataset(img_nchw, dataset)

    return preprocessing


def create_predict_function_pytorch(org_model, cps_model, device: torch.device):
    def predict(input_img: np.ndarray):
        input_tensor = torch.from_numpy(input_img).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            org_output = org_model(input_tensor).detach().cpu().numpy()
            cps_output = cps_model(input_tensor).detach().cpu().numpy()
        return PredictResult(org_output), PredictResult(cps_output)

    return predict


def validate_args(dataset: str, arch: str, cps_type: str):
    valid_arch = {
        "mnist": {"lenet1", "simplecnn", "lenet4", "lenet5"},
        "cifar": {"resnet", "resnet20", "plainnet20", "vgg16"},
    }
    valid_cps = {"quan", "prun", "kd"}

    if dataset not in valid_arch:
        raise ValueError("Only mnist and cifar are supported in the PyTorch pipeline.")
    if arch.lower() not in valid_arch[dataset]:
        raise ValueError(f"Unsupported arch '{arch}' for dataset '{dataset}'.")
    if cps_type.lower() not in valid_cps:
        raise ValueError("cps_type must be one of: quan, prun, kd.")


def get_seed_arch_name(dataset: str, arch: str) -> str:
    arch = arch.lower()
    if dataset == "mnist":
        seed_map = {
            "simplecnn": "lenet1",
        }
        return seed_map.get(arch, arch)
    if dataset == "cifar":
        seed_map = {
            "resnet20": "resnet",
        }
        return seed_map.get(arch, arch)
    return arch


def main():
    parser = common_argparser()
    parser.add_argument("--cps_type", choices=["quan", "prun", "kd"], required=True)
    args = parser.parse_args()

    validate_args(args.dataset, args.arch, args.cps_type)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Step 1: prepare output directory and logger.
    output_name = f"{args.arch}_{args.cps_type}_{args.seed}_{args.attack_mode}"
    save_dir = os.path.join(args.output_dir, f"{args.dataset}-pytorch", output_name)
    # Allow repeated runs with the same arguments.
    create_folder(save_dir, safe=False)
    logger_path = os.path.join(args.output_dir, f"{args.dataset}-pytorch", f"{output_name}.log")
    logger = myLogger.create_logger(logger_path)
    logger(args)

    # Step 2: load original/compressed PyTorch models.
    logger("Load PyTorch models")
    org_model, cps_model, org_path, cps_path = load_original_and_compressed_models(
        dataset=args.dataset,
        arch=args.arch,
        cps_type=args.cps_type,
        device=device,
    )
    logger(f"Original model: {org_path}")
    logger(f"Compressed model: {cps_path}")

    # Step 3: build prediction and preprocessing functions.
    predict_f = create_predict_function_pytorch(org_model, cps_model, device)
    preprocessing = build_preprocessing(args.dataset)

    # Step 4: load seed inputs from the original DFlare seed file.
    seed_arch = get_seed_arch_name(args.dataset, args.arch)
    input_sets = SubsetInputs(args.dataset, "tensorflow", seed_arch, "quan-lite", args.seed)

    # Step 5: run the original DFlare search logic.
    logger("Start the attack")
    tf_gen(args, input_sets, logger, save_dir, predict_f, preprocessing)


if __name__ == "__main__":
    main()
