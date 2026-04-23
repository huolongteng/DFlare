"""TFLite wrapper entrypoint for running DFlare differential testing.

This script wires together:
- model loading (original + compressed TFLite variants),
- input preprocessing,
- prediction adaptation into project result objects,
- and the main generation loop (``tf_gen``).
"""

import os  # Standard library: path composition and filesystem operations.
import numpy as np  # Third-party: numeric array ops for preprocessing.

from myLib.Inputs import SubsetInputs  # Dataset subset loader used as seed corpus.
from myUtils import create_folder, myLogger  # Utilities: mkdir helper + logger factory.
from test_gen_main import tf_gen  # Core differential-testing search loop.
from proj_utils import common_argparser  # Shared CLI argument definitions.
from model_loader.load_tflite import load_cps_model, load_org_model  # TFLite/H5 model loaders.
from myLib.Result import PredictResult  # Standardized prediction result wrapper.


def create_predict_function_tflite(org_model, cps_model):
    """Wrap two model callables into a unified prediction function.

    Args:
        org_model: Callable for original model inference.
        cps_model: Callable for compressed model inference.

    Returns:
        Callable: ``predict(input_img) -> (PredictResult, PredictResult)``.
    """

    def predict(input_img):
        # Forward pass through both models on the exact same preprocessed input.
        org_vec = org_model(input_img)
        cps_vec = cps_model(input_img)

        # Convert raw vectors to project-wide result objects
        # (label/prob/vector fields are accessed downstream).
        org_result = PredictResult(org_vec)
        cps_result = PredictResult(cps_vec)

        return org_result, cps_result

    return predict


def preprocessing_mnist_tflite(img: np.array) -> np.array:
    """Preprocess MNIST image to TFLite input format.

    Steps:
      1) cast to float32;
      2) ensure channel axis exists;
      3) convert CHW -> HWC when needed;
      4) add batch axis;
      5) normalize to [0, 1].
    """
    img = img.astype(np.float32)

    # If grayscale HxW, expand to HxWx1.
    if len(img.shape) == 2:
        img = img[..., np.newaxis]

    # If single image tensor (no batch dimension yet).
    if len(img.shape) == 3:
        # When input is CHW (1xHxW or 3xHxW), convert to HWC expected by TFLite.
        if img.shape[0] == 1 or img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))

        # Add batch dimension: HWC -> NHWC.
        img = img[np.newaxis, ...]

    # Scale pixel range from [0, 255] to [0, 1].
    img = img / 255.0

    return img


def preprocessing_cifar_tflite(img: np.array) -> np.array:
    """Preprocess CIFAR image to normalized NHWC float tensor for TFLite.

    Uses channel-wise mean/std normalization consistent with project settings.
    """
    img = img.astype(np.float32)

    # Add batch dimension if currently HWC.
    if len(img.shape) == 3:
        img = img[np.newaxis, ...]

    # Dataset-specific normalization parameters.
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]

    # Normalize each RGB channel independently.
    for i in range(3):
        img[:, :, :, i] = (img[:, :, :, i] - mean[i]) / std[i]

    return img


def main():
    """CLI entrypoint for running DFlare against TFLite compressed models."""
    parser = common_argparser()

    # Additional option specific to this wrapper.
    parser.add_argument("--cps_type", choices=["tflite", "quan"])
    args = parser.parse_args()

    # Basic consistency checks between dataset and architecture.
    assert args.dataset == "mnist" or args.dataset == "cifar"

    if "lenet1" in args.arch or "lenet5" in args.arch:
        assert args.dataset == "mnist"
    elif "resnet" in args.arch:
        assert args.dataset == "cifar"

    # Build output folder structure:
    # {output_dir}/{dataset}-tflite/{arch}_{cps_type}_{seed}_{attack_mode}/
    output_file_folder_name = "{}_{}_{}_{}".format(args.arch, args.cps_type, args.seed, args.attack_mode)
    save_dir = os.path.join(args.output_dir, "{}-tflite".format(args.dataset), output_file_folder_name)
    create_folder(save_dir)

    # Prepare logger file in dataset-level output folder.
    logger_filename = os.path.join(
        args.output_dir,
        "{}-tflite".format(args.dataset),
        "{}.log".format(output_file_folder_name),
    )
    logger = myLogger.create_logger(logger_filename)
    print(args)
    logger(args)

    logger("Load model")

    # Load original model.
    if "lenet1" in args.arch or "lenet5" in args.arch or "resnet" in args.arch:
        org_model_raw, org_model = load_org_model("./diffchaser_models/{}.h5".format(args.arch))
    else:
        raise NotImplemented

    # Load compressed model variant.
    if "tflite" in args.cps_type:
        cps_model_raw, cps_model = load_cps_model("./diffchaser_models/{}.lite".format(args.arch))
    elif "quan" in args.cps_type:
        cps_model_raw, cps_model = load_cps_model("./diffchaser_models/{}-quan.lite".format(args.arch))
    else:
        raise NotImplemented

    # Create prediction adapter used by the generation loop.
    predict_f = create_predict_function_tflite(org_model, cps_model)

    # Prepare seed input set.
    input_sets = SubsetInputs(args.dataset, "tensorflow", args.arch, "quan-lite", args.seed)

    logger("Stat the attack")

    # Select dataset-specific preprocessing.
    if args.dataset == "mnist":
        preprocessing = preprocessing_mnist_tflite
    elif args.dataset == "cifar":
        preprocessing = preprocessing_cifar_tflite
    else:
        raise NotImplementedError

    # Launch search.
    print("Start the attack")
    logger("Start the attack")
    tf_gen(args, input_sets, logger, save_dir, predict_f, preprocessing)


if __name__ == '__main__':
    """DFlare for compressed model using TFLite."""

    main()
