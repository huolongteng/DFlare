"""Project-level utility helpers used by DFlare entry scripts.

This module mainly provides:
1) a common CLI argument parser shared by different runners;
2) a summary formatter/logger for attack outcomes.
"""

import argparse  # Standard library: parse command-line arguments.
import numpy as np  # Third-party: numerical arrays and statistics.
import json  # Standard library: JSON handling (reserved for future extensions).

# Human-readable description for each attack mode code.
# These strings are used in summary logs so results are interpretable.
attack_modes = {
    "a": "DFlare",
    "b": "no MCMC guide mutation selection",
    "c": "random fitness function",
}


def common_argparser():
    """Create the shared argument parser for differential testing scripts.

    Returns:
        argparse.ArgumentParser: A configured parser containing dataset/model,
        search-budget, reproducibility, output, and mode-related arguments.

    Notes:
        - This parser intentionally only defines common arguments.
        - Script-specific runners may add extra arguments after calling it.
    """
    # Create parser with a short description shown in --help.
    parser = argparse.ArgumentParser("testing two DNN models")

    # Dataset to evaluate. Choices keep input valid and avoid silent typos.
    parser.add_argument(
        '--dataset',
        help="dataset",
        choices=['mnist', 'imagenet', 'cifar'],
        required=True,
    )

    # Model architecture name (e.g., lenet5/resnet20).
    parser.add_argument("--arch", help="model arch", default='resnet20')

    # Maximum mutation iterations per seed input.
    parser.add_argument("--maxit", type=int, help="max iteration", default=100)

    # Use first N seeds from the input corpus.
    parser.add_argument("--num", help="the first N will be used in the testing", default=1000, type=int)

    # Random seed for deterministic sampling/mutation when possible.
    parser.add_argument("--seed", help="random seed", default=0, type=int)

    # Output directory for logs and generated artifacts.
    parser.add_argument("--output_dir", help="output dir", default="./results", type=str)

    # Whether to force CPU mode (currently consumed by other modules).
    parser.add_argument("--cpu", help="use cpu for testing", default=False)

    # Search strategy mode.
    parser.add_argument(
        "--attack_mode",
        type=str,
        default="a",
        choices=list(attack_modes.keys()),
        help="a for DFlare, b for MCMC guide mutation selection, c for andom fitness function",
    )

    # Time budget (seconds) for each seed input search.
    parser.add_argument("--timeout", type=int, default=240, help="timeout for each seed input")

    return parser


def summary_attack_results(success_iter, logger, attack_mode):
    """Compute and print aggregate attack statistics.

    Args:
        success_iter (np.ndarray): Per-seed status/iteration array.
            Convention:
              -1 => failed within budget;
               0 => already divergent at seed (no search needed);
              >0 => found trigger at that iteration.
        logger (Callable[[str], None]): Project logger function.
        attack_mode (str): One of "a", "b", "c" used to decode mode name.
    """
    # Count each outcome category.
    no_search_rate = np.sum(success_iter == 0)
    success_rate = np.sum(success_iter > 0)
    failure_rate = np.sum(success_iter < 0)

    # Build summary text step-by-step for clear logging.
    summary_str = "Attack Summary for "
    summary_str += "Attack Mode: {}\n".format(attack_modes[attack_mode])
    summary_str += " Total {}, NoNeed {}, Success {}, Failed {}\n".format(
        len(success_iter),
        no_search_rate,
        success_rate,
        failure_rate,
    )

    # Only compute descriptive stats when there are successful seeds.
    success = success_iter[success_iter > 0]
    if len(success) > 0:
        summary_str = summary_str + "Avg {:.04f}".format(np.mean(success))
        summary_str = summary_str + "\tMedian {:.04f}".format(np.median(success))
        summary_str = summary_str + "\tMin {:04f}".format(np.min(success))
        summary_str = summary_str + "\tMax {:05f}".format(np.max(success))

    # Emit to stdout and persistent log file.
    print(summary_str)
    logger(summary_str)
