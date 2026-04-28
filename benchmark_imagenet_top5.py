import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from myLib.img_mutations import get_img_mutations
from myLib.probability_img_mutations import ProbabilityImgMutations, RandomImgMutations
from torchvision_quantized_imagenet_models import MODEL_SPECS, load_model


HF_REPO_ID = "ronanhansel/imagenet-1k-validation-subsets"
HF_FILENAME = "data/validation_1k-00000-of-00001.parquet"


def parse_args():
    parser = argparse.ArgumentParser("ImageNet top-5 benchmark for official torchvision quantized models.")
    parser.add_argument("--arch", choices=sorted(MODEL_SPECS.keys()), required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maxit", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-sample DFlare timeout in seconds.")
    parser.add_argument("--data-dir", type=str, default="./data/ImageNet-HF")
    parser.add_argument("--output-dir", type=str, default="./results_imagenet_top5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--attack-mode", choices=["prob", "random"], default="prob")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str):
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def download_validation_subset(data_dir: str):
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=HF_FILENAME,
        local_dir=data_dir,
    )


def load_samples(data_dir: str, num: int, seed: int):
    parquet_path = download_validation_subset(data_dir)
    frame = pd.read_parquet(parquet_path)
    if num < len(frame):
        frame = frame.sample(n=num, random_state=seed).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)

    samples = []
    for idx, row in frame.iterrows():
        image_cell = row["image"]
        image_bytes = image_cell["bytes"] if isinstance(image_cell, dict) else image_cell
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        samples.append({
            "idx": int(idx),
            "label": int(row["label"]),
            "image": image,
        })
    return samples, parquet_path


def get_transforms(arch: str):
    spec = MODEL_SPECS[arch]
    return spec["original_weights"].transforms(), spec["quantized_weights"].transforms()


def pil_to_uint8(image: Image.Image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def uint8_to_pil(image_np: np.ndarray):
    return Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8), mode="RGB")


def batch_predict(images, org_model, cps_model, org_transform, cps_transform, org_device):
    org_batch = torch.stack([org_transform(img) for img in images]).to(org_device)
    cps_batch = torch.stack([cps_transform(img) for img in images])
    with torch.no_grad():
        org_logits = org_model(org_batch).detach().cpu()
        cps_logits = cps_model(cps_batch).detach().cpu()
    return org_logits, cps_logits


def predict_one(image, org_model, cps_model, org_transform, cps_transform, org_device):
    org_logits, cps_logits = batch_predict(
        [image], org_model, cps_model, org_transform, cps_transform, org_device
    )
    return org_logits[0], cps_logits[0]


def topk_set(logits: torch.Tensor, k: int = 5):
    return set(torch.topk(logits, k=k).indices.tolist())


def topk_label(logits: torch.Tensor, k: int = 5):
    return torch.topk(logits, k=k).indices.tolist()


def top5_disagrees(org_logits: torch.Tensor, cps_logits: torch.Tensor):
    return topk_set(org_logits, 5) != topk_set(cps_logits, 5)


def top5_fitness(org_logits: torch.Tensor, cps_logits: torch.Tensor):
    org_probs = torch.softmax(org_logits.float(), dim=0)
    cps_probs = torch.softmax(cps_logits.float(), dim=0)
    org_top = topk_set(org_logits, 5)
    cps_top = topk_set(cps_logits, 5)
    union = sorted(org_top | cps_top)
    symdiff = len(org_top ^ cps_top)
    union_gap = torch.abs(org_probs[union] - cps_probs[union]).sum().item() if union else 0.0
    return float(symdiff + union_gap)


def evaluate_natural(samples, org_model, cps_model, org_transform, cps_transform, org_device, batch_size=16):
    records = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [item["image"] for item in batch]
        org_logits, cps_logits = batch_predict(images, org_model, cps_model, org_transform, cps_transform, org_device)
        for local_idx, item in enumerate(batch):
            label = item["label"]
            org_top5 = topk_label(org_logits[local_idx], 5)
            cps_top5 = topk_label(cps_logits[local_idx], 5)
            records.append({
                "idx": item["idx"],
                "label": label,
                "org_top1": int(org_top5[0]),
                "cps_top1": int(cps_top5[0]),
                "org_top5": org_top5,
                "cps_top5": cps_top5,
                "org_top1_correct": int(org_top5[0] == label),
                "cps_top1_correct": int(cps_top5[0] == label),
                "org_top5_correct": int(label in org_top5),
                "cps_top5_correct": int(label in cps_top5),
                "top5_disagreement": int(set(org_top5) != set(cps_top5)),
            })
    return records


def run_dflare(samples, org_model, cps_model, org_transform, cps_transform, org_device, args):
    records = []
    mutation_ops = get_img_mutations()

    for sample_idx, sample in enumerate(samples):
        raw_np = pil_to_uint8(sample["image"])
        org_logits, cps_logits = predict_one(
            sample["image"], org_model, cps_model, org_transform, cps_transform, org_device
        )
        start_time = time.time()
        best_score = top5_fitness(org_logits, cps_logits)

        if top5_disagrees(org_logits, cps_logits):
            records.append({
                "idx": sample_idx,
                "label": sample["label"],
                "success": 1,
                "queries": 0,
                "time_sec": time.time() - start_time,
                "iteration": 0,
                "best_score": best_score,
                "org_top5": topk_label(org_logits, 5),
                "cps_top5": topk_label(cps_logits, 5),
            })
            continue

        if args.attack_mode == "random":
            mutator_picker = RandomImgMutations(mutation_ops, args.seed + sample_idx)
        else:
            mutator_picker = ProbabilityImgMutations(mutation_ops, args.seed + sample_idx)

        latest_np = np.copy(raw_np)
        last_mutation_operator = None
        success = False
        queries = 0
        final_org_logits = org_logits
        final_cps_logits = cps_logits
        final_iteration = -1

        for iteration in range(1, args.maxit + 1):
            if time.time() - start_time > args.timeout:
                break
            mutator = mutator_picker.choose_mutator(last_mutation_operator)
            mutator.total += 1
            candidate_np = mutator.mut(np.copy(latest_np))
            candidate_pil = uint8_to_pil(candidate_np)
            cand_org_logits, cand_cps_logits = predict_one(
                candidate_pil, org_model, cps_model, org_transform, cps_transform, org_device
            )
            queries += 1

            score = top5_fitness(cand_org_logits, cand_cps_logits)
            if top5_disagrees(cand_org_logits, cand_cps_logits):
                mutator.delta_bigger_than_zero += 1
                success = True
                final_org_logits = cand_org_logits
                final_cps_logits = cand_cps_logits
                best_score = score
                final_iteration = iteration
                break

            if score > best_score:
                mutator.delta_bigger_than_zero += 1
                best_score = score
                latest_np = np.copy(candidate_np)
                last_mutation_operator = mutator.name
                final_org_logits = cand_org_logits
                final_cps_logits = cand_cps_logits

        records.append({
            "idx": sample_idx,
            "label": sample["label"],
            "success": int(success),
            "queries": queries,
            "time_sec": time.time() - start_time,
            "iteration": final_iteration,
            "best_score": best_score,
            "org_top5": topk_label(final_org_logits, 5),
            "cps_top5": topk_label(final_cps_logits, 5),
        })
    return records


def summarize(arch, num, seed, natural_records, dflare_records, data_path):
    org_top1 = float(np.mean([r["org_top1_correct"] for r in natural_records]))
    cps_top1 = float(np.mean([r["cps_top1_correct"] for r in natural_records]))
    org_top5 = float(np.mean([r["org_top5_correct"] for r in natural_records]))
    cps_top5 = float(np.mean([r["cps_top5_correct"] for r in natural_records]))
    natural_disagree = float(np.mean([r["top5_disagreement"] for r in natural_records]))
    success = int(sum(r["success"] for r in dflare_records))
    return {
        "method": "dflare_top5",
        "dataset": "imagenet_validation_1k_subset",
        "data_path": str(data_path),
        "arch": arch,
        "display_name": MODEL_SPECS[arch]["display_name"],
        "cps_type": "official_quantized",
        "num": num,
        "seed": seed,
        "original_top1_acc": org_top1,
        "quantized_top1_acc": cps_top1,
        "original_top5_acc": org_top5,
        "quantized_top5_acc": cps_top5,
        "natural_top5_disagreement_rate": natural_disagree,
        "success": success,
        "failed": len(dflare_records) - success,
        "success_rate": success / len(dflare_records) if dflare_records else 0.0,
        "avg_queries": float(np.mean([r["queries"] for r in dflare_records])) if dflare_records else 0.0,
        "avg_time_sec": float(np.mean([r["time_sec"] for r in dflare_records])) if dflare_records else 0.0,
    }


def write_outputs(output_dir, arch, seed, natural_records, dflare_records, summary):
    output_root = Path(output_dir)
    run_dir = output_root / f"{arch}_official_quant_top5_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "natural_records.json", "w", encoding="utf-8") as handle:
        json.dump(natural_records, handle, indent=2)
    with open(run_dir / "dflare_records.json", "w", encoding="utf-8") as handle:
        json.dump(dflare_records, handle, indent=2)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(run_dir / "dflare_records.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dflare_records[0].keys()))
        writer.writeheader()
        writer.writerows(dflare_records)

    print(json.dumps(summary, indent=2))
    return run_dir


def main():
    args = parse_args()
    set_seed(args.seed)
    org_device = choose_device(args.device)
    samples, data_path = load_samples(args.data_dir, args.num, args.seed)
    org_transform, cps_transform = get_transforms(args.arch)

    org_model = load_model(args.arch, "original").to(org_device).eval()
    cps_model = load_model(args.arch, "quantized").eval()

    natural_records = evaluate_natural(samples, org_model, cps_model, org_transform, cps_transform, org_device)
    dflare_records = run_dflare(samples, org_model, cps_model, org_transform, cps_transform, org_device, args)
    summary = summarize(args.arch, len(samples), args.seed, natural_records, dflare_records, data_path)
    write_outputs(args.output_dir, args.arch, args.seed, natural_records, dflare_records, summary)


if __name__ == "__main__":
    main()
