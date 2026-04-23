import argparse
import numpy as np
import json

attack_modes = {"a": "DFlare",
                "b": "no MCMC guide mutation selection",
                "c": "random fitness function"}


def common_argparser():
    parser = argparse.ArgumentParser("testing two DNN models")
    parser.add_argument('--dataset', help="dataset", choices=['mnist', 'imagenet', 'cifar'],
                        required=True)
    parser.add_argument("--arch", help="model arch", default='resnet20')

    parser.add_argument("--maxit", type=int, help="max iteration", default=100)
    parser.add_argument("--num", help="the first N will be used in the testing", default=1000, type=int)

    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--output_dir", help="output dir", default="./results", type=str)

    parser.add_argument("--cpu", help="use cpu for testing", default=False)

    parser.add_argument("--attack_mode", type=str, default="a", choices=list(attack_modes.keys()),
                        help="a for DFlare, b for MCMC guide mutation selection, c for andom fitness function")

    parser.add_argument("--timeout", type=int, default=240,
                        help="timeout for each seed input")


    return parser


def summary_attack_results(success_iter, logger, attack_mode):
    no_search_rate = np.sum(success_iter == 0)
    success_rate = np.sum(success_iter > 0)
    failure_rate = np.sum(success_iter < 0)
    total = len(success_iter)
    summary_str = "Attack Summary\n"
    summary_str += " Attack Mode: {}\n".format(attack_modes[attack_mode])
    summary_str += " Total Seeds: {}\n".format(total)
    summary_str += " NoNeed: {} ({:.2%})  # original/compressed already disagree on seed input\n".format(
        no_search_rate, no_search_rate / total
    )
    summary_str += " Success: {} ({:.2%}) # found disagreement after mutation\n".format(
        success_rate, success_rate / total
    )
    summary_str += " Failed: {} ({:.2%})  # no disagreement found within limits\n".format(
        failure_rate, failure_rate / total
    )

    success = success_iter[success_iter > 0]
    if len(success) > 0:
        summary_str = summary_str + " Iterations for successful cases -> Avg {:.04f}".format(np.mean(success))
        summary_str = summary_str + ", Median {:.04f}".format(np.median(success))
        summary_str = summary_str + ", Min {:04f}".format(np.min(success))
        summary_str = summary_str + ", Max {:05f}".format(np.max(success))

    print(summary_str)
    logger(summary_str)

