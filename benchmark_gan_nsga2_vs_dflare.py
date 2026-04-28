import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


DATASETS = {
    "mnist": ["lenet-4", "lenet-5", "simplecnn"],
    "cifar": ["plainnet20", "resnet20", "vgg16"],
}
CPS_TYPES = ["kd", "quant", "prune"]


DFLARE_TOTAL_RE = re.compile(r"Total\s+(\d+),\s+NoNeed\s+(\d+),\s+Success\s+(\d+),\s+Failed\s+(\d+)")
DFLARE_AVG_RE = re.compile(r"Avg\s+([0-9.]+)")
DFLARE_TIME_RE = re.compile(r"Time Summary\s+Avg\s+([0-9.]+)s")
DFLARE_RUNTIME_RE = re.compile(r"Total Runtime\s+([0-9.]+)s")


def parse_args():
    parser = argparse.ArgumentParser("Run DFlare vs GAN NSGA-II v2 benchmark grid.")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-dir", type=str, default="./results_gan_nsga2_vs_dflare_200x5")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dflare-maxit", type=int, default=1000)
    parser.add_argument("--dflare-attack-mode", choices=["a", "b", "c", "d"], default="a")
    parser.add_argument("--dflare-coverage-threshold", type=float, default=0.50)
    parser.add_argument("--gan-profile", choices=["manual", "auto"], default="manual",
                        help="manual uses the explicit GAN flags; auto uses tuned per-pair settings.")
    parser.add_argument("--gan-population-size", type=int, default=32)
    parser.add_argument("--gan-generations", type=int, default=8)
    parser.add_argument("--gan-batch-size", type=int, default=4096)
    parser.add_argument("--gan-chunk-size", type=int, default=0)
    parser.add_argument("--gan-fitness", choices=["adopted-euclidean", "conflict-margin", "divergence"],
                        default="adopted-euclidean")
    parser.add_argument("--gan-adaptive-budget", action="store_true")
    parser.add_argument("--gan-budget-stages", type=str, default="8x2,16x4,32x8")
    parser.add_argument("--datasets", type=str, default="mnist,cifar")
    parser.add_argument("--cps-types", type=str, default="kd,quant,prune")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--methods", choices=["both", "dflare", "gan"], default="both")
    return parser.parse_args()


def resolve_gan_config(args, dataset: str, arch: str, cps_type: str):
    config = {
        "population_size": args.gan_population_size,
        "generations": args.gan_generations,
        "batch_size": args.gan_batch_size,
        "chunk_size": args.gan_chunk_size,
        "fitness": args.gan_fitness,
        "adaptive_budget": args.gan_adaptive_budget,
        "budget_stages": args.gan_budget_stages,
    }
    if args.gan_profile == "manual":
        return config

    config.update({
        "population_size": 32,
        "generations": 8,
        "chunk_size": 4,
        "fitness": "adopted-euclidean",
        "adaptive_budget": True,
        "budget_stages": "8x2,16x4,32x8",
    })

    # Quantized pairs are less tolerant of the staged budget. Keep the full
    # evolutionary budget, but still use chunked evaluation so reported queries
    # better match a sequential black-box setting.
    if cps_type == "quant" and not (dataset == "mnist" and arch == "lenet-5"):
        config["adaptive_budget"] = False

    return config


def run_command(command, cwd: Path):
    start = time.time()
    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")
    return proc.stdout, elapsed


def parse_dflare_stdout(stdout: str):
    total_match = DFLARE_TOTAL_RE.search(stdout)
    avg_match = DFLARE_AVG_RE.search(stdout)
    time_match = DFLARE_TIME_RE.search(stdout)
    runtime_match = DFLARE_RUNTIME_RE.search(stdout)
    if not total_match:
        raise ValueError(f"Unable to parse DFlare summary:\n{stdout[-2000:]}")

    total, no_need, success, failed = map(int, total_match.groups())
    return {
        "method": "dflare",
        "total": total,
        "no_need": no_need,
        "success": success,
        "failed": failed,
        "success_total": no_need + success,
        "success_rate": (no_need + success) / total if total else 0.0,
        "avg_queries_reported": float(avg_match.group(1)) if avg_match else 0.0,
        "avg_time_sec": float(time_match.group(1)) if time_match else 0.0,
        "total_runtime_sec": float(runtime_match.group(1)) if runtime_match else 0.0,
    }


def gan_output_name(arch: str, cps_type: str, seed: int, pop: int, gen: int,
                    fitness: str, chunk_size: int, adaptive: bool, budget_stages: str):
    name = f"{arch}_{cps_type}_{seed}_gan_nsga2_pop{pop}_gen{gen}_{fitness}_js"
    if chunk_size > 0:
        name += f"_chunk{chunk_size}"
    if adaptive:
        safe_stages = budget_stages.replace(",", "-").replace("x", "x")
        name += f"_adaptive_{safe_stages}"
    return name


def load_gan_summary(output_root: Path, dataset: str, arch: str, cps_type: str, seed: int,
                     pop: int, gen: int, fitness: str, chunk_size: int,
                     adaptive: bool, budget_stages: str):
    summary_path = (
        output_root
        / f"{dataset}-torch"
        / gan_output_name(arch, cps_type, seed, pop, gen, fitness, chunk_size, adaptive, budget_stages)
        / "summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "method": "gan_nsga2_v2",
        "total": int(data["num"]),
        "no_need": 0,
        "success": int(data["success"]),
        "failed": int(data["failed"]),
        "success_total": int(data["success"]),
        "success_rate": float(data["success_rate"]),
        "avg_queries_reported": float(data["avg_queries"]),
        "avg_time_sec": float(data["avg_time_sec"]),
        "total_runtime_sec": float(data["total_runtime_sec"]),
    }


def write_rows(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row["method"], row["dataset"], row["arch"], row["cps_type"])
        groups.setdefault(key, []).append(row)

    out = []
    for (method, dataset, arch, cps_type), items in sorted(groups.items()):
        out.append({
            "method": method,
            "dataset": dataset,
            "arch": arch,
            "cps_type": cps_type,
            "runs": len(items),
            "success_rate_mean": sum(float(x["success_rate"]) for x in items) / len(items),
            "success_total_mean": sum(float(x["success_total"]) for x in items) / len(items),
            "avg_queries_mean": sum(float(x["avg_queries_reported"]) for x in items) / len(items),
            "avg_time_sec_mean": sum(float(x["avg_time_sec"]) for x in items) / len(items),
            "total_runtime_sec_mean": sum(float(x["total_runtime_sec"]) for x in items) / len(items),
        })
    return out


def main():
    args = parse_args()
    root = Path.cwd()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    cps_types = [x.strip() for x in args.cps_types.split(",") if x.strip()]

    rows = []
    python = sys.executable

    for dataset in datasets:
        for arch in DATASETS[dataset]:
            for cps_type in cps_types:
                for seed in seeds:
                    common = {
                        "dataset": dataset,
                        "arch": arch,
                        "cps_type": cps_type,
                        "seed": seed,
                        "num": args.num,
                        "gan_profile": args.gan_profile,
                        "dflare_attack_mode": args.dflare_attack_mode,
                        "dflare_coverage_threshold": args.dflare_coverage_threshold,
                    }

                    if args.methods in ("both", "dflare"):
                        dflare_cmd = [
                            python, "torch_dflare_local.py",
                            "--dataset", dataset,
                            "--arch", arch,
                            "--cps-type", cps_type,
                            "--num", str(args.num),
                            "--seed", str(seed),
                            "--maxit", str(args.dflare_maxit),
                            "--attack-mode", args.dflare_attack_mode,
                            "--coverage-threshold", str(args.dflare_coverage_threshold),
                            "--timeout", str(args.timeout),
                            "--output-dir", str(output_root / "dflare"),
                            "--device", args.device,
                        ]
                        print("RUN", " ".join(dflare_cmd))
                        stdout, wall = run_command(dflare_cmd, root)
                        result = parse_dflare_stdout(stdout)
                        result.update(common)
                        result["wall_time_sec"] = wall
                        rows.append(result)
                        write_rows(output_root / "raw_results.csv", rows)

                    if args.methods in ("both", "gan"):
                        gan_config = resolve_gan_config(args, dataset, arch, cps_type)
                        gan_cmd = [
                            python, "torch_gan_nsga2_search.py",
                            "--dataset", dataset,
                            "--arch", arch,
                            "--cps-type", cps_type,
                            "--num", str(args.num),
                            "--population-size", str(gan_config["population_size"]),
                            "--generations", str(gan_config["generations"]),
                            "--batch-size", str(gan_config["batch_size"]),
                            "--fitness", gan_config["fitness"],
                            "--seed", str(seed),
                            "--timeout", str(args.timeout),
                            "--output-dir", str(output_root / "gan_nsga2_v2"),
                            "--device", args.device,
                        ]
                        if gan_config["chunk_size"] > 0:
                            gan_cmd.extend(["--chunk-size", str(gan_config["chunk_size"])])
                        if gan_config["adaptive_budget"]:
                            gan_cmd.extend(["--adaptive-budget", "--budget-stages", gan_config["budget_stages"]])
                        print("RUN", " ".join(gan_cmd))
                        stdout, wall = run_command(gan_cmd, root)
                        result = load_gan_summary(
                            output_root / "gan_nsga2_v2",
                            dataset,
                            arch,
                            cps_type,
                            seed,
                            gan_config["population_size"],
                            gan_config["generations"],
                            gan_config["fitness"],
                            gan_config["chunk_size"],
                            gan_config["adaptive_budget"],
                            gan_config["budget_stages"],
                        )
                        result.update(common)
                        result["wall_time_sec"] = wall
                        rows.append(result)
                        write_rows(output_root / "raw_results.csv", rows)
                        write_rows(output_root / "aggregate_results.csv", aggregate(rows))

    write_rows(output_root / "raw_results.csv", rows)
    write_rows(output_root / "aggregate_results.csv", aggregate(rows))
    print(f"Wrote {output_root / 'raw_results.csv'}")
    print(f"Wrote {output_root / 'aggregate_results.csv'}")


if __name__ == "__main__":
    main()
