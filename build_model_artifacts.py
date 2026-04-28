import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import gan_model_utils
import torch_compress_models as compressor
from warning_utils import suppress_known_runtime_warnings


suppress_known_runtime_warnings()

def parse_args():
    parser = argparse.ArgumentParser("Build original, compressed, and GAN model artifacts.")
    parser.add_argument("--dataset", choices=["mnist", "fashion"], required=True)
    parser.add_argument("--arch", default="all")
    parser.add_argument("--target", choices=["originals", "compressed", "gan", "all", "inventory"], default="all")
    parser.add_argument("--models-dir", default="./models_")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument("--compression-method", choices=["kd", "quant", "prune", "all"], default="all")
    parser.add_argument("--kd-epochs", type=int, default=4)
    parser.add_argument("--finetune-epochs", type=int, default=3)
    parser.add_argument("--calib-samples", type=int, default=1024)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--prune-ratio", type=float, default=0.30)
    parser.add_argument("--kd-prune-ratio", type=float, default=0.50)
    parser.add_argument("--gan-epochs", type=int, default=5)
    parser.add_argument("--gan-lr", type=float, default=2e-4)
    parser.add_argument("--gan-batch-size", type=int, default=128)
    parser.add_argument("--gan-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def arch_list(dataset: str, arch: str):
    arches = list(compressor.ORIGINAL_MODEL_PATHS[dataset].keys())
    if arch == "all":
        return arches
    return [compressor.canonical_arch(dataset, arch)]


def output_root(models_dir: str, dataset: str):
    root = Path(models_dir) / compressor.DATASET_DIRS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_train_model(dataset: str, arch: str):
    return compressor.build_original_model(dataset, arch)


def build_classifier_loaders(args, dataset: str, device: torch.device):
    return compressor.build_dataloaders(
        dataset=dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        calib_samples=args.calib_samples,
        seed=args.seed,
        device=device,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    return compressor.evaluate_accuracy(model, loader, device)


def train_classifier(model, train_loader, test_loader, device, epochs: int, lr: float, weight_decay: float):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

        acc = evaluate(model, test_loader, device)
        epoch_report = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(total, 1),
            "test_accuracy": acc,
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report))
    return history


def save_original_artifact(args, dataset: str, arch: str, model, history, test_loader, device):
    root = output_root(args.models_dir, dataset)
    path = root / f"{arch}.pth"
    report_path = root / f"{arch}_original.json"
    model.cpu().eval()
    torch.save(model.state_dict(), path)
    model.to(device)
    report = {
        "dataset": dataset,
        "dataset_dir": compressor.DATASET_DIRS[dataset],
        "arch": arch,
        "artifact": str(path),
        "epochs": args.epochs,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "accuracy": evaluate(model, test_loader, device),
        "param_count": compressor.parameter_count(model),
        "state_tensor_bytes": compressor.state_dict_tensor_bytes(model),
        "saved_state_dict_bytes": path.stat().st_size,
        "history": history,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved original {dataset}/{arch} to {path}")
    return report


def build_originals(args, dataset: str, arches: list[str], device: torch.device):
    train_loader, calib_loader, test_loader = build_classifier_loaders(args, dataset, device)
    reports = []
    for arch in arches:
        root = output_root(args.models_dir, dataset)
        target = root / f"{arch}.pth"
        if target.exists() and not args.overwrite:
            print(f"Skipping existing original {target}")
            continue
        model = build_train_model(dataset, arch)
        history = train_classifier(model, train_loader, test_loader, device, args.epochs, args.lr, args.weight_decay)
        reports.append(save_original_artifact(args, dataset, arch, model, history, test_loader, device))
    return reports


def build_compressed(args, dataset: str, arches: list[str], device: torch.device):
    reports = []
    for arch in arches:
        checkpoint = output_root(args.models_dir, dataset) / f"{arch}.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Original checkpoint not found for compression: {checkpoint}")

        source_model = compressor.load_source_model(dataset, arch, str(checkpoint), device)
        train_loader, calib_loader, test_loader = build_classifier_loaders(args, dataset, device)
        baseline_acc = compressor.evaluate_accuracy(source_model, test_loader, device)
        base_args = SimpleNamespace(
            dataset=dataset,
            arch=arch,
            checkpoint=str(checkpoint),
            output_dir=args.models_dir,
            kd_epochs=args.kd_epochs,
            kd_lr=args.lr,
            kd_alpha=0.7,
            kd_temperature=4.0,
            kd_prune_ratio=args.kd_prune_ratio,
            calib_batches=args.calib_batches,
            calib_mode="dataset",
            calib_noise_scale=1.0,
            finetune_epochs=args.finetune_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            prune_ratio=args.prune_ratio,
            distill_alpha=0.5,
            distill_temperature=4.0,
            quant_backend="onednn",
        )
        methods = ["kd", "quant", "prune"] if args.compression_method == "all" else [args.compression_method]
        arch_reports = []
        for method in methods:
            report = compressor.run_method(
                base_args,
                method,
                source_model,
                train_loader,
                calib_loader,
                test_loader,
                baseline_acc,
                compressor.parameter_count(source_model),
                compressor.state_dict_tensor_bytes(source_model),
                device,
            )
            reports.append(report)
            arch_reports.append(report)
        if len(arch_reports) > 1:
            root = output_root(args.models_dir, dataset)
            with open(root / f"{arch}_compression.json", "w", encoding="utf-8") as handle:
                json.dump(arch_reports, handle, indent=2)
    return reports


class DCGANDiscriminator(nn.Module):
    def __init__(self, nc: int, ndf: int = 64, image_size: int = 28):
        super().__init__()
        layers = []
        if image_size == 64:
            layers.extend([
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            ])
        else:
            layers.extend([
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, 1, 3 if image_size == 28 else 4, 1, 0, bias=False),
            ])
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).reshape(-1)


def gan_transform(dataset: str):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_gan_dataset(args, dataset: str):
    transform = gan_transform(dataset)
    if dataset == "mnist":
        train_set = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    else:
        train_set = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform)
    return maybe_subset(train_set, args.gan_samples, args.seed + 10)


def train_gan(args, dataset: str, device: torch.device):
    image_size = 28
    nc = 1
    generator = gan_model_utils.DCGANGenerator(nc=nc, image_size=image_size).to(device)
    discriminator = DCGANDiscriminator(nc=nc, image_size=image_size).to(device)

    data = build_gan_dataset(args, dataset)
    loader = DataLoader(data, batch_size=args.gan_batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    fixed_noise = torch.randn(16, generator.nz, 1, 1, device=device)

    root = Path(args.models_dir) / "GAN" / compressor.DATASET_DIRS[dataset]
    root.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.gan_epochs):
        g_loss_total = 0.0
        d_loss_total = 0.0
        total = 0
        for real, _ in loader:
            real = real.to(device)
            batch_size = real.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            discriminator.zero_grad(set_to_none=True)
            real_loss = criterion(discriminator(real), real_targets)
            noise = torch.randn(batch_size, generator.nz, 1, 1, device=device)
            fake = generator(noise)
            fake_loss = criterion(discriminator(fake.detach()), fake_targets)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            opt_d.step()

            generator.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward()
            opt_g.step()

            g_loss_total += g_loss.item() * batch_size
            d_loss_total += d_loss.item() * batch_size
            total += batch_size

        epoch_report = {
            "epoch": epoch + 1,
            "g_loss": g_loss_total / max(total, 1),
            "d_loss": d_loss_total / max(total, 1),
        }
        history.append(epoch_report)
        print(json.dumps(epoch_report))
        torch.save(generator.state_dict(), root / f"dcgan_netG_epoch_{epoch + 1}.pth")

    latest_path = root / "dcgan_netG_latest.pth"
    torch.save(generator.state_dict(), latest_path)
    with torch.no_grad():
        samples = (generator(fixed_noise).detach().cpu() + 1.0) / 2.0
    torch.save(samples, root / "dcgan_fixed_samples.pt")
    manifest = {
        "dataset": dataset,
        "dataset_dir": compressor.DATASET_DIRS[dataset],
        "generator": str(latest_path),
        "epochs": args.gan_epochs,
        "gan_samples": args.gan_samples,
        "history": history,
        "saved_at_unix": time.time(),
    }
    with open(root / "dcgan_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved GAN generator to {latest_path}")
    return manifest


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def artifact_entry(path: Path):
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def collect_inventory(models_dir: str, dataset: str):
    root = output_root(models_dir, dataset)
    inventory = {
        "dataset": dataset,
        "dataset_dir": compressor.DATASET_DIRS[dataset],
        "models_root": str(root),
        "architectures": {},
        "gan": {},
    }

    for arch in compressor.ORIGINAL_MODEL_PATHS[dataset]:
        arch_info = {
            "original": artifact_entry(root / f"{arch}.pth"),
            "original_report": read_json(root / f"{arch}_original.json"),
            "compressed": {},
            "compression_summary": read_json(root / f"{arch}_compression.json"),
        }
        for method in ("kd", "quant", "prune"):
            arch_info["compressed"][method] = {
                "model": artifact_entry(root / f"{arch}_{method}.pt"),
                "state_dict": artifact_entry(root / f"{arch}_{method}.pth"),
                "report": read_json(root / f"{arch}_{method}.json"),
            }
        inventory["architectures"][arch] = arch_info

    gan_root = Path(models_dir) / "GAN" / compressor.DATASET_DIRS[dataset]
    inventory["gan"] = {
        "root": str(gan_root),
        "generator": artifact_entry(gan_root / "dcgan_netG_latest.pth"),
        "manifest": read_json(gan_root / "dcgan_manifest.json"),
        "fixed_samples": artifact_entry(gan_root / "dcgan_fixed_samples.pt"),
    }
    with open(root / "artifact_inventory.json", "w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)
    return inventory


def main():
    args = parse_args()
    set_seed(args.seed)
    device = compressor.resolve_device(args.device)
    arches = arch_list(args.dataset, args.arch)

    all_reports = {"originals": [], "compressed": [], "gan": None}
    if args.target in ("originals", "all"):
        all_reports["originals"] = build_originals(args, args.dataset, arches, device)
    if args.target in ("compressed", "all"):
        all_reports["compressed"] = build_compressed(args, args.dataset, arches, device)
    if args.target in ("gan", "all"):
        all_reports["gan"] = train_gan(args, args.dataset, device)

    root = output_root(args.models_dir, args.dataset)
    if args.target != "inventory":
        with open(root / "build_manifest.json", "w", encoding="utf-8") as handle:
            json.dump(all_reports, handle, indent=2)
    inventory = collect_inventory(args.models_dir, args.dataset)
    print(f"Wrote inventory to {root / 'artifact_inventory.json'}")
    print(json.dumps({
        "dataset": inventory["dataset"],
        "architectures": list(inventory["architectures"].keys()),
        "gan_exists": inventory["gan"]["generator"]["exists"],
    }, indent=2))


if __name__ == "__main__":
    main()
