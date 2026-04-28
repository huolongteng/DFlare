import argparse
import json
from pathlib import Path

from diffusers import DDPMPipeline


MODEL_SPECS = {
    "mnist": {
        "repo_id": "1aurent/ddpm-mnist",
        "local_name": "mnist-ddpm",
        "description": "Unconditional DDPM trained on MNIST.",
    },
    "cifar": {
        "repo_id": "google/ddpm-cifar10-32",
        "local_name": "cifar10-ddpm",
        "description": "Unconditional DDPM trained on CIFAR-10 32x32 images.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Download pretrained DDPM models for local probing.")
    parser.add_argument("--dataset", choices=["mnist", "cifar", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default="./models_/DIFFUSION")
    parser.add_argument("--force", action="store_true", help="Download again even if the local folder exists.")
    return parser.parse_args()


def selected_specs(dataset):
    if dataset == "all":
        return MODEL_SPECS.items()
    return [(dataset, MODEL_SPECS[dataset])]


def model_looks_saved(path: Path) -> bool:
    return (path / "model_index.json").exists() and (path / "scheduler" / "scheduler_config.json").exists()


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for dataset, spec in selected_specs(args.dataset):
        local_dir = output_root / spec["local_name"]
        if model_looks_saved(local_dir) and not args.force:
            print(f"[skip] {dataset}: already saved at {local_dir}")
        else:
            print(f"[download] {dataset}: {spec['repo_id']} -> {local_dir}")
            pipe = DDPMPipeline.from_pretrained(spec["repo_id"])
            pipe.save_pretrained(local_dir, safe_serialization=True)

        manifest[dataset] = {
            **spec,
            "local_dir": str(local_dir),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote {manifest_path}")


if __name__ == "__main__":
    main()
