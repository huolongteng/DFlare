import argparse
import json
from pathlib import Path

from torchvision import __version__ as torchvision_version
from torchvision import models
from torchvision.models import quantization as qmodels


DOC_URL = "https://docs.pytorch.org/vision/stable/models.html#quantized-models"


MODEL_SPECS = {
    "inception_v3": {
        "display_name": "Inception V3",
        "original_builder": models.inception_v3,
        "original_weights": models.Inception_V3_Weights.DEFAULT,
        "quantized_builder": qmodels.inception_v3,
        "quantized_weights": qmodels.Inception_V3_QuantizedWeights.DEFAULT,
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "original_builder": models.resnet50,
        "original_weights": models.ResNet50_Weights.DEFAULT,
        "quantized_builder": qmodels.resnet50,
        "quantized_weights": qmodels.ResNet50_QuantizedWeights.DEFAULT,
    },
    "resnext101_32x8d": {
        "display_name": "ResNeXt-101 32x8d",
        "original_builder": models.resnext101_32x8d,
        "original_weights": models.ResNeXt101_32X8D_Weights.DEFAULT,
        "quantized_builder": qmodels.resnext101_32x8d,
        "quantized_weights": qmodels.ResNeXt101_32X8D_QuantizedWeights.DEFAULT,
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Record torchvision ImageNet original/quantized model metadata.")
    parser.add_argument("--output-dir", default="./models_/ImageNet-Quantized")
    parser.add_argument("--eval-output-dir", default="./compression_eval")
    return parser.parse_args()


def metric(weights, key: str):
    return weights.meta["_metrics"]["ImageNet-1K"][key]


def model_size_mb(weights):
    return weights.meta["_file_size"]


def build_rows():
    rows = []
    for arch, spec in MODEL_SPECS.items():
        original = spec["original_weights"]
        quantized = spec["quantized_weights"]
        original_size = model_size_mb(original)
        quantized_size = model_size_mb(quantized)
        original_top1 = metric(original, "acc@1")
        quantized_top1 = metric(quantized, "acc@1")
        original_top5 = metric(original, "acc@5")
        quantized_top5 = metric(quantized, "acc@5")
        rows.append({
            "arch": arch,
            "model": spec["display_name"],
            "original_weights": f"{original.__class__.__name__}.{original.name}",
            "quantized_weights": f"{quantized.__class__.__name__}.{quantized.name}",
            "params": original.meta["num_params"],
            "original_top1": original_top1,
            "quantized_top1": quantized_top1,
            "top1_delta": quantized_top1 - original_top1,
            "original_top5": original_top5,
            "quantized_top5": quantized_top5,
            "top5_delta": quantized_top5 - original_top5,
            "original_file_mb": original_size,
            "quantized_file_mb": quantized_size,
            "file_ratio": quantized_size / original_size,
            "original_url": original.url,
            "quantized_url": quantized.url,
            "original_recipe": original.meta.get("recipe"),
            "quantized_recipe": quantized.meta.get("recipe"),
        })
    return rows


def format_markdown(rows):
    lines = [
        "# ImageNet Torchvision Quantized Models",
        "",
        f"Source: [{DOC_URL}]({DOC_URL})",
        f"Torchvision version used for metadata: `{torchvision_version}`",
        "",
        "| Model | Original Weights | Quantized Weights | Top-1 Original | Top-1 Quant | Delta | Top-5 Original | Top-5 Quant | Size MB Original | Size MB Quant | Size Ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | `{original_weights}` | `{quantized_weights}` | "
            "{original_top1:.3f} | {quantized_top1:.3f} | {top1_delta:+.3f} | "
            "{original_top5:.3f} | {quantized_top5:.3f} | "
            "{original_file_mb:.3f} | {quantized_file_mb:.3f} | {file_ratio:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows, output_dir: str, eval_output_dir: str):
    output_root = Path(output_dir)
    eval_root = Path(eval_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": DOC_URL,
        "torchvision_version": torchvision_version,
        "note": "These entries reuse torchvision's official ImageNet-1K pretrained and quantized weights; no local KD, pruning, or fine-tuning is applied.",
        "rows": rows,
    }
    markdown = format_markdown(rows)

    with open(output_root / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(output_root / "comparison.md", "w", encoding="utf-8") as handle:
        handle.write(markdown)
    with open(eval_root / "imagenet_quantized_comparison.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(eval_root / "imagenet_quantized_comparison.md", "w", encoding="utf-8") as handle:
        handle.write(markdown)
    return markdown


def load_model(arch: str, variant: str = "quantized"):
    spec = MODEL_SPECS[arch]
    if variant == "original":
        model = spec["original_builder"](weights=spec["original_weights"])
    elif variant == "quantized":
        model = spec["quantized_builder"](weights=spec["quantized_weights"], quantize=True)
    else:
        raise ValueError("variant must be 'original' or 'quantized'")
    return model.eval()


def main():
    args = parse_args()
    rows = build_rows()
    markdown = write_outputs(rows, args.output_dir, args.eval_output_dir)
    print(markdown)


if __name__ == "__main__":
    main()
