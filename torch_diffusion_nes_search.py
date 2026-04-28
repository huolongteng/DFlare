import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from myUtils import create_folder, myLogger
from torch_diffusion_cem_search import (
    batched_generate,
    diffusion_shape,
    expand_low_res_noise,
    low_res_side,
)
from torch_diffusion_random_search import (
    divergence_scores,
    generated_to_uint8,
    load_diffusion_pipeline,
    save_success_image,
)
from torch_model_utils import canonical_arch, load_models, predict_batch
from warning_utils import suppress_known_runtime_warnings


suppress_known_runtime_warnings()


def parse_args():
    parser = argparse.ArgumentParser("NES-guided diffusion search for original + compressed models")
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--arch", required=True,
                        help="mnist: lenet-4, lenet-5, simplecnn; cifar: resnet20, plainnet20, vgg16")
    parser.add_argument("--cps-type", choices=["kd", "quant", "prune"], default="kd")
    parser.add_argument("--steps", type=int, default=5, help="Diffusion denoising steps.")
    parser.add_argument("--num", type=int, default=100, help="Number of independent NES trials.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results_diffusion_nes")
    parser.add_argument("--timeout", type=int, default=240, help="Overall timeout in seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--diffusion-models-dir", type=str, default="./models_/DIFFUSION")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--score", choices=["js", "l1", "l2", "margin"], default="js")
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--sigma-decay", type=float, default=0.95)
    parser.add_argument("--low-res", type=int, default=0,
                        help="Low-res noise side length. 0 uses 7 for MNIST and 8 for CIFAR.")
    parser.add_argument("--antithetic", action="store_true",
                        help="Use paired +epsilon/-epsilon samples. Population size should be even.")
    parser.add_argument("--save-success-images", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def normalized_utilities(scores: torch.Tensor):
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def write_outputs(save_dir: Path, summary: dict, records: list[dict]):
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (save_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def run_nes(args, logger, save_dir: Path):
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

    if args.antithetic and args.population_size % 2 != 0:
        raise ValueError("--population-size must be even when --antithetic is used.")

    channels, height, width = diffusion_shape(args.dataset)
    low_side = low_res_side(args)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    center = torch.zeros(args.num, channels, low_side, low_side, device=device, dtype=dtype)
    active = torch.ones(args.num, dtype=torch.bool, device=device)
    queries = torch.zeros(args.num, dtype=torch.int32, device=device)
    success_generation = torch.full((args.num,), -1, dtype=torch.int32, device=device)
    success_score = torch.full((args.num,), float("-inf"), dtype=torch.float32, device=device)
    success_org_label = torch.full((args.num,), -1, dtype=torch.int32, device=device)
    success_cps_label = torch.full((args.num,), -1, dtype=torch.int32, device=device)
    best_score = torch.full((args.num,), float("-inf"), dtype=torch.float32, device=device)
    trial_times = torch.zeros(args.num, dtype=torch.float32, device=device)

    records = []
    success_img_dir = save_dir / "success_images"
    if args.save_success_images:
        success_img_dir.mkdir(parents=True, exist_ok=True)

    current_sigma = args.sigma
    for generation in range(1, args.generations + 1):
        if time.time() - overall_start > args.timeout:
            logger("Time Out")
            break

        gen_start = time.time()
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break

        active_center = center[active_indices]
        if args.antithetic:
            half = args.population_size // 2
            eps_half = torch.randn(
                (active_indices.numel(), half, channels, low_side, low_side),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            eps = torch.cat([eps_half, -eps_half], dim=1)
        else:
            eps = torch.randn(
                (active_indices.numel(), args.population_size, channels, low_side, low_side),
                generator=generator,
                device=device,
                dtype=dtype,
            )

        candidates = active_center[:, None] + current_sigma * eps
        flat_candidates = candidates.reshape(-1, channels, low_side, low_side)
        initial_noise = expand_low_res_noise(flat_candidates, height, width)

        generated = batched_generate(pipe, initial_noise, args.steps, args.batch_size)
        org_logits, cps_logits = predict_batch(args.dataset, generated, org_model, cps_model, org_device, cps_device)
        scores = divergence_scores(org_logits, cps_logits, args.score).reshape(active_indices.numel(), args.population_size)

        org_probs = F.softmax(org_logits.float(), dim=1)
        cps_probs = F.softmax(cps_logits.float(), dim=1)
        org_labels = torch.argmax(org_probs, dim=1).reshape(active_indices.numel(), args.population_size)
        cps_labels = torch.argmax(cps_probs, dim=1).reshape(active_indices.numel(), args.population_size)
        success_mask = org_labels != cps_labels

        scores_device = scores.to(device)
        success_mask_device = success_mask.to(device)
        org_labels_device = org_labels.to(device)
        cps_labels_device = cps_labels.to(device)

        row_best_score, _ = torch.max(scores_device, dim=1)
        best_score[active_indices] = torch.maximum(best_score[active_indices], row_best_score)

        query_add = torch.full((active_indices.numel(),), args.population_size, dtype=torch.int32, device=device)
        row_has_success = success_mask_device.any(dim=1)
        if row_has_success.any():
            first_success = torch.argmax(success_mask_device.int(), dim=1).to(torch.int32)
            query_add[row_has_success] = first_success[row_has_success] + 1
            successful_trials = active_indices[row_has_success]
            successful_offsets = first_success[row_has_success].long()

            success_generation[successful_trials] = generation
            success_score[successful_trials] = scores_device[row_has_success, successful_offsets]
            success_org_label[successful_trials] = org_labels_device[row_has_success, successful_offsets].int()
            success_cps_label[successful_trials] = cps_labels_device[row_has_success, successful_offsets].int()

            if args.save_success_images:
                generated_uint8 = generated_to_uint8(args.dataset, generated)
                for local_row, global_idx in zip(torch.nonzero(row_has_success, as_tuple=False).flatten().tolist(),
                                                 successful_trials.tolist()):
                    offset = int(first_success[local_row].item())
                    flat_idx = local_row * args.population_size + offset
                    image_path = success_img_dir / (
                        f"{global_idx:06d}_gen{generation}_"
                        f"org{int(org_labels[local_row, offset].item())}_"
                        f"cps{int(cps_labels[local_row, offset].item())}.png"
                    )
                    save_success_image(args.dataset, generated_uint8[flat_idx], image_path)

        queries[active_indices] += query_add

        not_success_rows = ~row_has_success
        if not_success_rows.any():
            surviving_trials = active_indices[not_success_rows]
            surviving_scores = scores_device[not_success_rows]
            surviving_eps = eps[not_success_rows].float()
            utilities = normalized_utilities(surviving_scores).float()
            direction = torch.mean(
                utilities[:, :, None, None, None] * surviving_eps,
                dim=1,
            )
            center[surviving_trials] = center[surviving_trials] + args.lr * direction.to(dtype)

        active[active_indices[row_has_success]] = False
        gen_elapsed = time.time() - gen_start
        trial_times[active_indices] += gen_elapsed / max(1, active_indices.numel())
        current_sigma *= args.sigma_decay
        logger(
            f"Generation {generation}/{args.generations}; "
            f"active {int(active.sum().item())}/{args.num}; "
            f"success {int((success_generation > 0).sum().item())}; "
            f"sigma {current_sigma:.4f}"
        )

    total_runtime = time.time() - overall_start
    success_cpu = (success_generation.cpu().numpy() > 0)
    queries_cpu = queries.cpu().numpy()
    trial_times_cpu = trial_times.cpu().numpy()
    best_score_cpu = best_score.cpu().numpy()
    success_generation_cpu = success_generation.cpu().numpy()
    success_score_cpu = success_score.cpu().numpy()
    success_org_cpu = success_org_label.cpu().numpy()
    success_cps_cpu = success_cps_label.cpu().numpy()

    for idx in range(args.num):
        records.append({
            "idx": idx,
            "success": int(success_cpu[idx]),
            "queries": int(queries_cpu[idx]),
            "time_sec": float(trial_times_cpu[idx]),
            "success_generation": int(success_generation_cpu[idx]),
            "success_score": float(success_score_cpu[idx]) if success_cpu[idx] else "",
            "best_score": float(best_score_cpu[idx]),
            "org_label": int(success_org_cpu[idx]) if success_cpu[idx] else "",
            "cps_label": int(success_cps_cpu[idx]) if success_cpu[idx] else "",
        })

    success_count = int(success_cpu.sum())
    failed = args.num - success_count
    avg_queries = float(np.mean(queries_cpu))
    avg_time = float(np.mean(trial_times_cpu))
    summary = {
        "method": "diffusion_nes_search",
        "dataset": args.dataset,
        "arch": args.arch,
        "cps_type": args.cps_type,
        "num": args.num,
        "success": success_count,
        "failed": failed,
        "success_rate": success_count / args.num if args.num else 0.0,
        "avg_queries": avg_queries,
        "avg_time_sec": avg_time,
        "total_runtime_sec": total_runtime,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "population_size": args.population_size,
        "generations": args.generations,
        "lr": args.lr,
        "sigma": args.sigma,
        "sigma_decay": args.sigma_decay,
        "low_res": low_side,
        "score": args.score,
        "antithetic": args.antithetic,
        "scheduler": pipe.scheduler.__class__.__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "diffusion_model_dir": str(diffusion_model_dir),
    }

    summary_str = (
        "Attack Summary for Diffusion NES Search\n"
        f" Total {args.num}, Success {success_count}, Failed {failed}\n"
        f"Success Rate {summary['success_rate']:.4f}\tAvg Queries {avg_queries:.4f}\tAvg Time {avg_time:.4f}s\n"
        f"Total Runtime {total_runtime:.4f}s"
    )
    print(summary_str)
    logger(summary_str)
    write_outputs(save_dir, summary, records)


def main():
    args = parse_args()
    arch = canonical_arch(args.dataset, args.arch)
    output_name = (
        f"{arch}_{args.cps_type}_{args.seed}_nes_"
        f"steps{args.steps}_pop{args.population_size}_gen{args.generations}_"
        f"lr{args.lr}_sigma{args.sigma}_{args.score}"
    )
    if args.antithetic:
        output_name += "_anti"
    save_dir = Path(args.output_dir) / f"{args.dataset}-torch" / output_name
    create_folder(str(save_dir))

    log_path = Path(args.output_dir) / f"{args.dataset}-torch" / f"{output_name}.log"
    logger = myLogger.create_logger(str(log_path))
    run_nes(args, logger, save_dir)


if __name__ == "__main__":
    main()
