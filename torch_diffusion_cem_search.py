import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from myUtils import create_folder, myLogger
from torch_model_utils import canonical_arch, load_models, predict_batch
from torch_diffusion_random_search import (
    divergence_scores,
    generated_to_uint8,
    load_diffusion_pipeline,
    save_success_image,
)
from warning_utils import suppress_known_runtime_warnings


suppress_known_runtime_warnings()


def parse_args():
    parser = argparse.ArgumentParser("CEM-guided diffusion search for original + compressed models")
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--arch", required=True,
                        help="mnist: lenet-4, lenet-5, simplecnn; cifar: resnet20, plainnet20, vgg16")
    parser.add_argument("--cps-type", choices=["kd", "quant", "prune"], default="kd")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion denoising steps.")
    parser.add_argument("--num", type=int, default=100, help="Number of independent CEM trials.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results_diffusion_cem")
    parser.add_argument("--timeout", type=int, default=240, help="Overall timeout in seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--diffusion-models-dir", type=str, default="./models_/DIFFUSION")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scheduler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16"], default="auto")
    parser.add_argument("--score", choices=["js", "l1", "l2", "margin"], default="js")
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--low-res", type=int, default=0,
                        help="Low-res noise side length. 0 uses 7 for MNIST and 8 for CIFAR.")
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--save-success-images", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def diffusion_shape(dataset: str):
    if dataset == "mnist":
        return 1, 28, 28
    return 3, 32, 32


def low_res_side(args):
    if args.low_res > 0:
        return args.low_res
    return 7 if args.dataset == "mnist" else 8


def expand_low_res_noise(low_noise: torch.Tensor, height: int, width: int):
    full = F.interpolate(low_noise, size=(height, width), mode="bilinear", align_corners=False)
    flat = full.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (full - mean) / std


def generate_from_initial_noise(pipe, initial_noise: torch.Tensor, steps: int):
    pipe.scheduler.set_timesteps(steps, device=initial_noise.device)
    sample = initial_noise * pipe.scheduler.init_noise_sigma
    with torch.inference_mode():
        for timestep in pipe.scheduler.timesteps:
            model_output = pipe.unet(sample, timestep).sample
            sample = pipe.scheduler.step(model_output, timestep, sample).prev_sample
    sample = (sample / 2 + 0.5).clamp(0, 1)
    sample = sample.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return sample


def batched_generate(pipe, initial_noise: torch.Tensor, steps: int, batch_size: int):
    outputs = []
    for start in range(0, initial_noise.shape[0], batch_size):
        chunk = initial_noise[start:start + batch_size]
        outputs.append(generate_from_initial_noise(pipe, chunk, steps))
    return np.concatenate(outputs, axis=0)


def write_outputs(save_dir: Path, summary: dict, records: list[dict]):
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (save_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def run_cem(args, logger, save_dir: Path):
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

    channels, height, width = diffusion_shape(args.dataset)
    low_side = low_res_side(args)
    elite_count = max(1, int(round(args.population_size * args.elite_frac)))
    generator = torch.Generator(device=device).manual_seed(args.seed)

    mu = torch.zeros(args.num, channels, low_side, low_side, device=device, dtype=dtype)
    sigma = torch.ones_like(mu)
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

    for generation in range(1, args.generations + 1):
        if time.time() - overall_start > args.timeout:
            logger("Time Out")
            break

        gen_start = time.time()
        active_indices = torch.nonzero(active, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            break

        active_mu = mu[active_indices]
        active_sigma = sigma[active_indices]
        noise_shape = (active_indices.numel(), args.population_size, channels, low_side, low_side)
        eps = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
        candidates = active_mu[:, None] + active_sigma[:, None] * eps
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
            surviving_candidates = candidates[not_success_rows]
            elite_idx = torch.topk(surviving_scores, k=elite_count, dim=1).indices
            gather_idx = elite_idx[:, :, None, None, None].expand(-1, -1, channels, low_side, low_side)
            elites = torch.gather(surviving_candidates, dim=1, index=gather_idx)
            mu[surviving_trials] = elites.mean(dim=1)
            sigma[surviving_trials] = elites.std(dim=1, unbiased=False).clamp_min(args.sigma_min)

        active[active_indices[row_has_success]] = False
        gen_elapsed = time.time() - gen_start
        trial_times[active_indices] += gen_elapsed / max(1, active_indices.numel())
        logger(
            f"Generation {generation}/{args.generations}; "
            f"active {int(active.sum().item())}/{args.num}; "
            f"success {int((success_generation > 0).sum().item())}"
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
        "method": "diffusion_cem_search",
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
        "elite_frac": args.elite_frac,
        "elite_count": elite_count,
        "low_res": low_side,
        "score": args.score,
        "scheduler": pipe.scheduler.__class__.__name__,
        "dtype": str(dtype).replace("torch.", ""),
        "diffusion_model_dir": str(diffusion_model_dir),
    }

    summary_str = (
        "Attack Summary for Diffusion CEM Search\n"
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
        f"{arch}_{args.cps_type}_{args.seed}_cem_"
        f"steps{args.steps}_pop{args.population_size}_gen{args.generations}_{args.score}"
    )
    save_dir = Path(args.output_dir) / f"{args.dataset}-torch" / output_name
    create_folder(str(save_dir))

    log_path = Path(args.output_dir) / f"{args.dataset}-torch" / f"{output_name}.log"
    logger = myLogger.create_logger(str(log_path))
    run_cem(args, logger, save_dir)


if __name__ == "__main__":
    main()
