import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMPipeline

from myUtils import create_folder, myLogger
from torch_model_utils import canonical_arch, load_models, predict_batch
from warning_utils import suppress_known_runtime_warnings


suppress_known_runtime_warnings()


DIFFUSION_MODEL_DIRS = {
    "mnist": "mnist-ddpm",
    "cifar": "cifar10-ddpm",
}


def parse_args():
    parser = argparse.ArgumentParser("Diffusion random-search runner for original + compressed models")
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--arch", required=True,
                        help="mnist: lenet-4, lenet-5, simplecnn; cifar: resnet20, plainnet20, vgg16")
    parser.add_argument("--cps-type", choices=["kd", "quant", "prune"], default="kd")
    parser.add_argument("--steps", type=int, default=50, help="Diffusion denoising steps.")
    parser.add_argument("--num", type=int, default=100, help="Number of generated candidates to test.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results_diffusion_random")
    parser.add_argument("--timeout", type=int, default=240, help="Overall timeout in seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--diffusion-models-dir", type=str, default="./models_/DIFFUSION")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--score", choices=["js", "l1", "l2", "margin"], default="js")
    parser.add_argument("--save-success-images", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def choose_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    return torch.float16 if device.type == "cuda" else torch.float32


def load_diffusion_pipeline(args, device: torch.device):
    model_dir = Path(args.diffusion_models_dir) / DIFFUSION_MODEL_DIRS[args.dataset]
    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} does not exist. Run download_diffusion_models.py first."
        )

    dtype = choose_dtype(args.dtype, device)
    pipe = DDPMPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    if args.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()

    if device.type == "cuda":
        pipe.unet.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    return pipe, dtype, model_dir


def divergence_scores(org_logits: torch.Tensor, cps_logits: torch.Tensor, score_name: str):
    org_prob = F.softmax(org_logits.float(), dim=1)
    cps_prob = F.softmax(cps_logits.float(), dim=1)

    if score_name == "l1":
        return torch.sum(torch.abs(org_prob - cps_prob), dim=1)
    if score_name == "l2":
        return torch.linalg.vector_norm(org_prob - cps_prob, ord=2, dim=1)
    if score_name == "margin":
        org_label = torch.argmax(org_prob, dim=1)
        cps_label = torch.argmax(cps_prob, dim=1)
        org_conf = org_prob.gather(1, org_label[:, None]).squeeze(1)
        cps_conf = cps_prob.gather(1, cps_label[:, None]).squeeze(1)
        org_on_cps = org_prob.gather(1, cps_label[:, None]).squeeze(1)
        cps_on_org = cps_prob.gather(1, org_label[:, None]).squeeze(1)
        return (org_conf - org_on_cps) + (cps_conf - cps_on_org)

    midpoint = 0.5 * (org_prob + cps_prob)
    eps = 1e-12
    org_kl = torch.sum(org_prob * (torch.log(org_prob + eps) - torch.log(midpoint + eps)), dim=1)
    cps_kl = torch.sum(cps_prob * (torch.log(cps_prob + eps) - torch.log(midpoint + eps)), dim=1)
    return 0.5 * (org_kl + cps_kl)


def generated_to_uint8(dataset: str, images_np: np.ndarray):
    images_np = np.asarray(images_np)
    if images_np.max() <= 1.0:
        images_np = images_np * 255.0
    images_np = np.clip(images_np, 0, 255).astype(np.uint8)
    if dataset == "mnist" and images_np.ndim == 3:
        images_np = images_np[..., np.newaxis]
    return images_np


def save_success_image(dataset: str, image_np: np.ndarray, path: Path):
    from PIL import Image

    if dataset == "mnist":
        if image_np.ndim == 3:
            image_np = image_np[..., 0]
        Image.fromarray(image_np, mode="L").save(path)
    else:
        Image.fromarray(image_np, mode="RGB").save(path)


def write_summary(save_dir: Path, summary: dict, records: list[dict]):
    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    records_path = save_dir / "records.csv"
    if records:
        with records_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def run_search(args, logger, save_dir: Path):
    overall_start = time.time()
    device = choose_device(args.device)
    args.arch = canonical_arch(args.dataset, args.arch)

    logger(args)
    logger("Load models")
    org_model, cps_model, org_device, cps_device = load_models(args.dataset, args.arch, args.cps_type, device)
    logger(f"Using org device {org_device}; compressed model device {cps_device}")

    logger("Load diffusion pipeline")
    pipe, dtype, diffusion_model_dir = load_diffusion_pipeline(args, device)
    logger(f"Diffusion model {diffusion_model_dir}; scheduler {pipe.scheduler.__class__.__name__}; dtype {dtype}")

    successes = 0
    records = []
    success_iters = np.ones([args.num], dtype=np.float32) * -1
    sample_times = np.zeros([args.num], dtype=np.float32)
    next_index = 0

    success_img_dir = save_dir / "success_images"
    if args.save_success_images:
        success_img_dir.mkdir(parents=True, exist_ok=True)

    while next_index < args.num:
        if time.time() - overall_start > args.timeout:
            logger("Time Out")
            break

        current_batch = min(args.batch_size, args.num - next_index)
        batch_start = time.time()
        generated = pipe(
            batch_size=current_batch,
            generator=torch.Generator(device=device).manual_seed(args.seed + next_index),
            num_inference_steps=args.steps,
            output_type="np",
        ).images
        generated_uint8 = generated_to_uint8(args.dataset, generated)

        org_logits, cps_logits = predict_batch(
            args.dataset, generated, org_model, cps_model, org_device, cps_device
        )
        scores = divergence_scores(org_logits, cps_logits, args.score)
        org_probs = F.softmax(org_logits.float(), dim=1)
        cps_probs = F.softmax(cps_logits.float(), dim=1)
        org_labels = torch.argmax(org_probs, dim=1)
        cps_labels = torch.argmax(cps_probs, dim=1)
        org_conf = org_probs.gather(1, org_labels[:, None]).squeeze(1)
        cps_conf = cps_probs.gather(1, cps_labels[:, None]).squeeze(1)
        batch_elapsed = time.time() - batch_start
        per_sample_time = batch_elapsed / current_batch

        for offset in range(current_batch):
            idx = next_index + offset
            success = bool(org_labels[offset].item() != cps_labels[offset].item())
            if success:
                successes += 1
                success_iters[idx] = 1
                if args.save_success_images:
                    image_path = success_img_dir / f"{idx:06d}_org{org_labels[offset].item()}_cps{cps_labels[offset].item()}.png"
                    save_success_image(args.dataset, generated_uint8[offset], image_path)

            sample_times[idx] = per_sample_time
            records.append({
                "idx": idx,
                "success": int(success),
                "queries": 1,
                "time_sec": per_sample_time,
                "score": float(scores[offset].item()),
                "org_label": int(org_labels[offset].item()),
                "cps_label": int(cps_labels[offset].item()),
                "org_conf": float(org_conf[offset].item()),
                "cps_conf": float(cps_conf[offset].item()),
            })

        logger(f"Generated {next_index + current_batch}/{args.num}; successes {successes}")
        next_index += current_batch

    evaluated = len(records)
    failed = evaluated - successes
    total_runtime = time.time() - overall_start
    success_rate = successes / evaluated if evaluated else 0.0
    avg_queries = float(np.mean([r["queries"] for r in records])) if records else 0.0
    avg_time = float(np.mean(sample_times[:evaluated])) if evaluated else 0.0

    summary = {
        "method": "diffusion_random_search",
        "dataset": args.dataset,
        "arch": args.arch,
        "cps_type": args.cps_type,
        "num_requested": args.num,
        "num_evaluated": evaluated,
        "success": successes,
        "failed": failed,
        "success_rate": success_rate,
        "avg_queries": avg_queries,
        "avg_time_sec": avg_time,
        "total_runtime_sec": total_runtime,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "score": args.score,
        "scheduler": pipe.scheduler.__class__.__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "diffusion_model_dir": str(diffusion_model_dir),
    }

    summary_str = (
        "Attack Summary for Diffusion Random Search\n"
        f" Total {evaluated}, Success {successes}, Failed {failed}\n"
        f"Success Rate {success_rate:.4f}\tAvg Queries {avg_queries:.4f}\tAvg Time {avg_time:.4f}s\n"
        f"Total Runtime {total_runtime:.4f}s"
    )
    print(summary_str)
    logger(summary_str)
    write_summary(save_dir, summary, records)


def main():
    args = parse_args()
    arch = canonical_arch(args.dataset, args.arch)
    output_name = f"{arch}_{args.cps_type}_{args.seed}_steps{args.steps}_{args.score}"
    save_dir = Path(args.output_dir) / f"{args.dataset}-torch" / output_name
    create_folder(str(save_dir))

    log_path = Path(args.output_dir) / f"{args.dataset}-torch" / f"{output_name}.log"
    logger = myLogger.create_logger(str(log_path))
    run_search(args, logger, save_dir)


if __name__ == "__main__":
    main()
