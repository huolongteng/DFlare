import os

import numpy as np
import torch

from myLib.Inputs import SubsetInputs
from myLib.Result import PredictResult
from myUtils import create_folder, myLogger
from proj_utils import common_argparser
from test_gen_main import tf_gen

from model_loader.load_pytorch_models import load_model_pair, preprocess_seed_input


def create_predict_function_pytorch(org_model, cps_model, dataset, device):
    def predict(input_img):
        input_tensor = preprocess_seed_input(dataset, input_img).to(device)

        with torch.no_grad():
            org_logits = org_model(input_tensor).cpu().numpy()
            cps_logits = cps_model(input_tensor).cpu().numpy()

        org_result = PredictResult(org_logits)
        cps_result = PredictResult(cps_logits)
        return org_result, cps_result

    return predict


def identity_preprocessing(img: np.array) -> np.array:
    return img


def main():
    parser = common_argparser()
    parser.add_argument("--cps_type", choices=["quan", "prun", "kd"], required=True)
    parser.add_argument("--model_root", type=str, default="./models_")
    args = parser.parse_args()

    if args.arch in ["lenet1", "lenet5"] and args.dataset != "mnist":
        raise ValueError("lenet1/lenet5 must be used with --dataset mnist")
    if args.arch == "resnet" and args.dataset != "cifar":
        raise ValueError("resnet must be used with --dataset cifar")
    if args.arch not in ["lenet1", "lenet5", "resnet"]:
        raise ValueError("Supported arch: lenet1, lenet5, resnet")

    use_cpu = str(args.cpu).lower() in ["1", "true", "yes"]
    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"

    output_name = f"{args.arch}_{args.cps_type}_{args.seed}_{args.attack_mode}"
    save_dir = os.path.join(args.output_dir, f"{args.dataset}-pytorch", output_name)
    create_folder(save_dir)

    log_file = os.path.join(args.output_dir, f"{args.dataset}-pytorch", f"{output_name}.log")
    logger = myLogger.create_logger(log_file)
    logger(args)
    logger(f"Using device: {device}")

    org_model, cps_model = load_model_pair(
        dataset=args.dataset,
        arch=args.arch,
        cps_type=args.cps_type,
        model_root=args.model_root,
        device=device,
    )

    predict_f = create_predict_function_pytorch(org_model, cps_model, args.dataset, device)

    # Reuse the existing seed pool and DFlare search logic.
    input_sets = SubsetInputs(args.dataset, "tensorflow", args.arch, args.cps_type, args.seed)

    print("Start the attack")
    logger("Start the attack")
    tf_gen(args, input_sets, logger, save_dir, predict_f, identity_preprocessing)


if __name__ == "__main__":
    main()
