import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gan_model_utils import gan_output_to_numpy, load_gan_generator
from myUtils import create_folder, myLogger
from torch_diffusion_random_search import divergence_scores, generated_to_uint8, save_success_image
from torch_model_utils import canonical_arch, load_models, predict_batch
from warning_utils import suppress_known_runtime_warnings


suppress_known_runtime_warnings()


def parse_args():
    parser = argparse.ArgumentParser("GAN latent NSGA-II search for original + compressed models")
    parser.add_argument("--dataset", choices=["mnist", "cifar"], required=True)
    parser.add_argument("--arch", required=True,
                        help="mnist: lenet-4, lenet-5, simplecnn; cifar: resnet20, plainnet20, vgg16")
    parser.add_argument("--cps-type", choices=["kd", "quant", "prune"], default="kd")
    parser.add_argument("--num", type=int, default=100, help="Number of independent NSGA-II trials.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./results_gan_nsga2")
    parser.add_argument("--timeout", type=int, default=240, help="Overall timeout in seconds.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gan-models-dir", type=str, default="./models_/GAN")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score", choices=["js", "l1", "l2", "margin"], default="js")
    parser.add_argument("--fitness", choices=["divergence", "adopted-euclidean", "conflict-margin"],
                        default="adopted-euclidean")
    parser.add_argument("--profile", choices=["manual", "auto"], default="manual",
                        help="manual uses explicit search flags; auto uses tuned per-pair GAN NSGA-II settings.")
    parser.add_argument("--init", choices=["uniform", "normal"], default="uniform")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Evaluate each generation in chunks for stricter early-stop queries. 0 uses full population.")
    parser.add_argument("--adaptive-budget", action="store_true",
                        help="Use a staged population/generation budget for easy-to-hard trials.")
    parser.add_argument("--budget-stages", type=str, default="8x2,16x4,32x8",
                        help="Adaptive stages formatted as popxgen comma list, e.g. 8x2,16x4,32x8.")
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-std", type=float, default=0.25)
    parser.add_argument("--latent-clip", type=float, default=1.0)
    parser.add_argument("--elite-frac", type=float, default=0.125)
    parser.add_argument("--save-success-images", action="store_true")
    return parser.parse_args()


def apply_profile(args):
    if args.profile != "auto":
        return args

    args.population_size = 32
    args.generations = 8
    args.chunk_size = 4
    args.fitness = "adopted-euclidean"
    args.adaptive_budget = True
    args.budget_stages = "8x2,16x4,32x8"

    if args.cps_type == "quant" and not (args.dataset == "mnist" and args.arch == "lenet-5"):
        args.adaptive_budget = False

    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def latent_diversity(population: torch.Tensor):
    distances = torch.cdist(population.float(), population.float(), p=2)
    eye = torch.eye(population.shape[0], device=population.device, dtype=torch.bool)
    distances = distances.masked_fill(eye, float("inf"))
    return torch.min(distances, dim=1).values


def adopted_euclidean_scores(org_logits: torch.Tensor, cps_logits: torch.Tensor):
    org_prob = F.softmax(org_logits.float(), dim=1)
    cps_prob = F.softmax(cps_logits.float(), dim=1)
    org_prob = org_prob / org_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cps_prob = cps_prob / cps_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    distance = torch.linalg.vector_norm(org_prob - cps_prob, ord=2, dim=1)
    disagreement_bonus = (torch.argmax(org_prob, dim=1) != torch.argmax(cps_prob, dim=1)).float()
    return distance + disagreement_bonus


def conflict_margin_scores(org_logits: torch.Tensor, cps_logits: torch.Tensor):
    org_prob = F.softmax(org_logits.float(), dim=1)
    cps_prob = F.softmax(cps_logits.float(), dim=1)
    org_label = torch.argmax(org_prob, dim=1)
    cps_label = torch.argmax(cps_prob, dim=1)
    org_conf = org_prob.gather(1, org_label[:, None]).squeeze(1)
    cps_conf = cps_prob.gather(1, cps_label[:, None]).squeeze(1)
    org_on_cps = org_prob.gather(1, cps_label[:, None]).squeeze(1)
    cps_on_org = cps_prob.gather(1, org_label[:, None]).squeeze(1)
    same_label_penalty = (org_label == cps_label).float()
    return (org_conf - org_on_cps) + (cps_conf - cps_on_org) - same_label_penalty


def nsga2_rank_and_crowding(objectives: torch.Tensor):
    obj = objectives.detach().cpu().numpy()
    n = obj.shape[0]
    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=np.int32)
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dominates_q = np.all(obj[p] >= obj[q]) and np.any(obj[p] > obj[q])
            q_dominates_p = np.all(obj[q] >= obj[p]) and np.any(obj[q] > obj[p])
            if p_dominates_q:
                dominates[p].append(q)
            elif q_dominates_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    rank = np.zeros(n, dtype=np.int32)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            rank[p] = i
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    crowding = np.zeros(n, dtype=np.float32)
    for front in fronts:
        if not front:
            continue
        front_obj = obj[front]
        if len(front) <= 2:
            crowding[front] = float("inf")
            continue
        for m in range(obj.shape[1]):
            order = np.argsort(front_obj[:, m])
            sorted_front = np.asarray(front)[order]
            crowding[sorted_front[0]] = float("inf")
            crowding[sorted_front[-1]] = float("inf")
            span = front_obj[order[-1], m] - front_obj[order[0], m]
            if span <= 1e-12:
                continue
            for j in range(1, len(front) - 1):
                crowding[sorted_front[j]] += (
                    front_obj[order[j + 1], m] - front_obj[order[j - 1], m]
                ) / span

    return rank, crowding


def tournament_indices(rank: np.ndarray, crowding: np.ndarray, count: int, rng: np.random.RandomState):
    selected = []
    n = len(rank)
    for _ in range(count):
        a, b = rng.randint(0, n, size=2)
        if rank[a] < rank[b]:
            selected.append(a)
        elif rank[b] < rank[a]:
            selected.append(b)
        elif crowding[a] >= crowding[b]:
            selected.append(a)
        else:
            selected.append(b)
    return np.asarray(selected, dtype=np.int64)


def make_next_population(population: torch.Tensor, scores: torch.Tensor, diversity: torch.Tensor,
                         args, rng: np.random.RandomState):
    pop_size, latent_dim = population.shape
    objectives = torch.stack([scores.float(), diversity.float()], dim=1)
    rank, crowding = nsga2_rank_and_crowding(objectives)
    order = sorted(range(pop_size), key=lambda i: (rank[i], -crowding[i]))
    elite_count = max(1, int(round(pop_size * args.elite_frac)))
    next_pop = [population[i].clone() for i in order[:elite_count]]

    while len(next_pop) < pop_size:
        parent_idx = tournament_indices(rank, crowding, 2, rng)
        p1 = population[int(parent_idx[0])]
        p2 = population[int(parent_idx[1])]
        if rng.rand() < args.crossover_rate:
            alpha = torch.rand(latent_dim, device=population.device, dtype=population.dtype)
            child = alpha * p1 + (1.0 - alpha) * p2
        else:
            child = p1.clone()

        mutation_mask = torch.rand(latent_dim, device=population.device) < args.mutation_rate
        if mutation_mask.any():
            noise = torch.randn(latent_dim, device=population.device, dtype=population.dtype) * args.mutation_std
            child = torch.where(mutation_mask, child + noise, child)
        if args.latent_clip > 0:
            child = child.clamp(-args.latent_clip, args.latent_clip)
        next_pop.append(child)

    return torch.stack(next_pop, dim=0)


def evaluate_latents(dataset: str, latents: torch.Tensor, gan, org_model, cps_model,
                     org_device: torch.device, cps_device: torch.device, batch_size: int,
                     score_name: str, fitness_name: str):
    all_scores = []
    all_org_labels = []
    all_cps_labels = []
    all_images = []

    with torch.inference_mode():
        for start in range(0, latents.shape[0], batch_size):
            z = latents[start:start + batch_size]
            generated = gan(z)
            images_np = gan_output_to_numpy(dataset, generated)
            org_logits, cps_logits = predict_batch(dataset, images_np, org_model, cps_model, org_device, cps_device)
            if fitness_name == "adopted-euclidean":
                scores = adopted_euclidean_scores(org_logits, cps_logits)
            elif fitness_name == "conflict-margin":
                scores = conflict_margin_scores(org_logits, cps_logits)
            else:
                scores = divergence_scores(org_logits, cps_logits, score_name)
            org_labels = torch.argmax(F.softmax(org_logits.float(), dim=1), dim=1)
            cps_labels = torch.argmax(F.softmax(cps_logits.float(), dim=1), dim=1)
            all_scores.append(scores)
            all_org_labels.append(org_labels)
            all_cps_labels.append(cps_labels)
            all_images.append(images_np)

    return (
        torch.cat(all_scores, dim=0),
        torch.cat(all_org_labels, dim=0),
        torch.cat(all_cps_labels, dim=0),
        np.concatenate(all_images, axis=0),
    )


def write_outputs(save_dir: Path, summary: dict, records: list[dict]):
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (save_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def parse_budget_stages(args):
    if not args.adaptive_budget:
        return [(args.population_size, args.generations)]

    stages = []
    for item in args.budget_stages.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid budget stage '{item}'. Use popxgen format.")
        pop_s, gen_s = item.split("x", 1)
        pop, gen = int(pop_s), int(gen_s)
        if pop <= 0 or gen <= 0:
            raise ValueError(f"Invalid budget stage '{item}'. Values must be positive.")
        stages.append((pop, gen))
    if not stages:
        raise ValueError("--adaptive-budget requires at least one --budget-stages entry.")
    return stages


def build_initial_population(num, pop_size, latent_dim, init, latent_clip, device, generator):
    if init == "uniform":
        population = (torch.rand(num, pop_size, latent_dim, generator=generator, device=device) * 2.0) - 1.0
    else:
        population = torch.randn(num, pop_size, latent_dim, generator=generator, device=device)
    if latent_clip > 0:
        population = population.clamp(-latent_clip, latent_clip)
    return population


def resize_population(population: torch.Tensor, new_pop_size: int, args, rng: np.random.RandomState):
    current_pop_size = population.shape[1]
    if current_pop_size == new_pop_size:
        return population
    if current_pop_size > new_pop_size:
        return population[:, :new_pop_size].contiguous()

    extra_count = new_pop_size - current_pop_size
    extras = []
    for trial_idx in range(population.shape[0]):
        pop = population[trial_idx]
        trial_extras = []
        while len(trial_extras) < extra_count:
            a, b = rng.randint(0, current_pop_size, size=2)
            alpha = torch.rand(args.latent_dim, device=population.device, dtype=population.dtype)
            child = alpha * pop[a] + (1.0 - alpha) * pop[b]
            mutation_mask = torch.rand(args.latent_dim, device=population.device) < args.mutation_rate
            if mutation_mask.any():
                noise = torch.randn(args.latent_dim, device=population.device, dtype=population.dtype) * args.mutation_std
                child = torch.where(mutation_mask, child + noise, child)
            if args.latent_clip > 0:
                child = child.clamp(-args.latent_clip, args.latent_clip)
            trial_extras.append(child)
        extras.append(torch.stack(trial_extras, dim=0))
    return torch.cat([population, torch.stack(extras, dim=0)], dim=1)


def run_nsga2(args, logger, save_dir: Path):
    overall_start = time.time()
    device = choose_device(args.device)
    args.arch = canonical_arch(args.dataset, args.arch)
    rng = np.random.RandomState(args.seed)
    torch_generator = torch.Generator(device=device).manual_seed(args.seed)

    logger(args)
    logger("Load models")
    org_model, cps_model, org_device, cps_device = load_models(args.dataset, args.arch, args.cps_type, device)
    logger(f"Using org device {org_device}; compressed model device {cps_device}")

    logger("Load GAN generator")
    gan, gan_path = load_gan_generator(args.dataset, device, args.gan_models_dir)
    logger(f"GAN generator {gan_path}")

    stages = parse_budget_stages(args)
    first_pop, _ = stages[0]
    max_pop = max(pop for pop, _ in stages)
    population = build_initial_population(
        args.num,
        first_pop,
        args.latent_dim,
        args.init,
        args.latent_clip,
        device,
        torch_generator,
    )

    active = np.ones(args.num, dtype=bool)
    queries = np.zeros(args.num, dtype=np.int32)
    success_generation = np.full(args.num, -1, dtype=np.int32)
    success_score = np.full(args.num, -np.inf, dtype=np.float32)
    success_org_label = np.full(args.num, -1, dtype=np.int32)
    success_cps_label = np.full(args.num, -1, dtype=np.int32)
    best_score = np.full(args.num, -np.inf, dtype=np.float32)
    trial_times = np.zeros(args.num, dtype=np.float32)

    success_img_dir = save_dir / "success_images"
    if args.save_success_images:
        success_img_dir.mkdir(parents=True, exist_ok=True)

    global_generation = 0
    for stage_idx, (stage_pop, stage_generations) in enumerate(stages, start=1):
        population = resize_population(population, stage_pop, args, rng)
        chunk_size = args.chunk_size if args.chunk_size > 0 else stage_pop
        chunk_size = max(1, min(chunk_size, stage_pop))

        for local_generation in range(1, stage_generations + 1):
            if time.time() - overall_start > args.timeout:
                logger("Time Out")
                break

            active_indices = np.flatnonzero(active)
            if len(active_indices) == 0:
                break

            global_generation += 1
            gen_start = time.time()
            score_chunks = []

            for chunk_start in range(0, stage_pop, chunk_size):
                current_active = np.flatnonzero(active)
                if len(current_active) == 0:
                    break
                chunk_end = min(chunk_start + chunk_size, stage_pop)
                chunk_latents = population[current_active, chunk_start:chunk_end].reshape(-1, args.latent_dim)
                scores, org_labels, cps_labels, images_np = evaluate_latents(
                    args.dataset,
                    chunk_latents,
                    gan,
                    org_model,
                    cps_model,
                    org_device,
                    cps_device,
                    args.batch_size,
                    args.score,
                    args.fitness,
                )
                current_chunk_size = chunk_end - chunk_start
                scores = scores.reshape(len(current_active), current_chunk_size)
                org_labels = org_labels.reshape(len(current_active), current_chunk_size)
                cps_labels = cps_labels.reshape(len(current_active), current_chunk_size)
                success_mask = org_labels != cps_labels
                score_chunks.append((current_active.copy(), chunk_start, chunk_end, scores.detach().cpu()))

                chunk_elapsed = time.time() - gen_start
                per_trial_elapsed = chunk_elapsed / max(1, len(current_active))
                for row, trial_idx in enumerate(current_active):
                    row_scores = scores[row]
                    row_success = success_mask[row]
                    best_score[trial_idx] = max(best_score[trial_idx], float(row_scores.max().item()))

                    if bool(row_success.any().item()):
                        first_success = int(torch.argmax(row_success.int()).item())
                        queries[trial_idx] += first_success + 1
                        success_generation[trial_idx] = global_generation
                        success_score[trial_idx] = float(row_scores[first_success].item())
                        success_org_label[trial_idx] = int(org_labels[row, first_success].item())
                        success_cps_label[trial_idx] = int(cps_labels[row, first_success].item())
                        trial_times[trial_idx] += per_trial_elapsed
                        active[trial_idx] = False

                        if args.save_success_images:
                            flat_idx = row * current_chunk_size + first_success
                            image_path = success_img_dir / (
                                f"{trial_idx:06d}_gen{global_generation}_"
                                f"org{success_org_label[trial_idx]}_cps{success_cps_label[trial_idx]}.png"
                            )
                            save_success_image(
                                args.dataset,
                                generated_to_uint8(args.dataset, images_np[flat_idx:flat_idx + 1])[0],
                                image_path,
                            )
                    else:
                        queries[trial_idx] += current_chunk_size

            gen_elapsed = time.time() - gen_start
            active_after_chunks = np.flatnonzero(active)
            for trial_idx in active_after_chunks:
                trial_times[trial_idx] += gen_elapsed / max(1, len(active_indices))

            row_scores_by_trial = {}
            for chunk_active, chunk_start, chunk_end, chunk_scores in score_chunks:
                for row, trial_idx in enumerate(chunk_active):
                    if trial_idx not in row_scores_by_trial:
                        row_scores_by_trial[trial_idx] = torch.empty(stage_pop)
                    row_scores_by_trial[trial_idx][chunk_start:chunk_end] = chunk_scores[row]

            for trial_idx in active_after_chunks:
                if trial_idx not in row_scores_by_trial:
                    continue
                row_scores = row_scores_by_trial[trial_idx]
                diversity = latent_diversity(population[trial_idx])
                population[trial_idx] = make_next_population(
                    population[trial_idx],
                    row_scores.to(device),
                    diversity,
                    args,
                    rng,
                )

            logger(
                f"Stage {stage_idx}/{len(stages)} generation {local_generation}/{stage_generations}; "
                f"pop {stage_pop}; active {int(active.sum())}/{args.num}; "
                f"success {int(np.sum(success_generation > 0))}"
            )

        if time.time() - overall_start > args.timeout or not active.any():
            break

    total_runtime = time.time() - overall_start
    success = success_generation > 0
    records = []
    for idx in range(args.num):
        records.append({
            "idx": idx,
            "success": int(success[idx]),
            "queries": int(queries[idx]),
            "time_sec": float(trial_times[idx]),
            "success_generation": int(success_generation[idx]),
            "success_score": float(success_score[idx]) if success[idx] else "",
            "best_score": float(best_score[idx]),
            "org_label": int(success_org_label[idx]) if success[idx] else "",
            "cps_label": int(success_cps_label[idx]) if success[idx] else "",
        })

    success_count = int(np.sum(success))
    failed = args.num - success_count
    avg_queries = float(np.mean(queries))
    avg_time = float(np.mean(trial_times))
    summary = {
        "method": "gan_nsga2_search",
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
        "population_size": args.population_size,
        "generations": args.generations,
        "chunk_size": args.chunk_size if args.chunk_size > 0 else args.population_size,
        "adaptive_budget": args.adaptive_budget,
        "budget_stages": stages,
        "max_population_size": max_pop,
        "latent_dim": args.latent_dim,
        "batch_size": args.batch_size,
        "score": args.score,
        "fitness": args.fitness,
        "profile": args.profile,
        "init": args.init,
        "crossover_rate": args.crossover_rate,
        "mutation_rate": args.mutation_rate,
        "mutation_std": args.mutation_std,
        "latent_clip": args.latent_clip,
        "elite_frac": args.elite_frac,
        "gan_path": str(gan_path),
    }

    summary_str = (
        "Attack Summary for GAN NSGA-II Search\n"
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
    args.arch = arch
    apply_profile(args)
    output_name = (
        f"{arch}_{args.cps_type}_{args.seed}_gan_nsga2_"
        f"pop{args.population_size}_gen{args.generations}_{args.fitness}_{args.score}"
    )
    if args.chunk_size > 0:
        output_name += f"_chunk{args.chunk_size}"
    if args.adaptive_budget:
        safe_stages = args.budget_stages.replace(",", "-").replace("x", "x")
        output_name += f"_adaptive_{safe_stages}"
    save_dir = Path(args.output_dir) / f"{args.dataset}-torch" / output_name
    create_folder(str(save_dir))

    log_path = Path(args.output_dir) / f"{args.dataset}-torch" / f"{output_name}.log"
    logger = myLogger.create_logger(str(log_path))
    run_nsga2(args, logger, save_dir)


if __name__ == "__main__":
    main()
