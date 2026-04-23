"""Core generation/search loop for DFlare-style differential testing.

This module exposes ``tf_gen`` which performs mutation-based search for each
seed input and tries to find a triggering input where two models disagree.
"""

import numpy as np  # Third-party: vector math, distance, and array containers.
import time  # Standard library: wall-clock timeout control.

from myLib.Result import SingleAttackResult  # Store and persist per-seed attack results.
from myLib.img_mutations import get_img_mutations  # Build the available mutation operator pool.
from myLib.covered_states import CoveredStates  # Coverage/state tracker used by mode a/b/d.
from proj_utils import summary_attack_results  # Final aggregate metrics and logging.


# attack mode reference:
#   a: default DFlare
#   b: disable probability-guided mutation selector (random selector)
#   c: disable state coverage fitness (use probability diff only)
#   d: variant with L2-weighted reward update


def tf_gen(args, inputs_set, logger, save_dir, predict_f, preprocessing):
    """Run differential testing over an input subset.

    Args:
        args: Parsed CLI args; requires maxit/timeout/attack_mode/seed, etc.
        inputs_set: Iterable seed dataset wrapper (dict with ``img`` and ``label``).
        logger: Logging callback.
        save_dir (str): Directory to write result artifacts.
        predict_f: Callable that takes preprocessed image and returns
            (org_result, cps_result).
        preprocessing: Callable that maps raw input image -> model-ready tensor.

    Returns:
        None. Results are saved via ``SingleAttackResult.save()`` and summary log.
    """
    # Total number of seed inputs to evaluate.
    number_of_data = inputs_set.len

    # Per-seed metadata arrays. Initialize with -1 to indicate failure by default.
    success_iter = np.ones([number_of_data]) * -1
    l2distances = np.ones([number_of_data]) * -1
    newl2distances = np.ones([number_of_data]) * -1
    num_reductions = []

    # Conventions:
    #   -1 => no trigger found within budget
    #    0 => already different prediction at seed (no mutation needed)
    #   >0 => iteration index where trigger is found

    for idx in range(0, number_of_data):
        # Load one seed sample.
        seed_file = inputs_set[idx]
        raw_seed_input = seed_file["img"]
        seed_label = seed_file["label"]

        logger("Img idx {} label {}".format(idx, seed_label))

        # Initialize coverage tracker when current attack mode uses state guidance.
        if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
            covered_states = CoveredStates()
        elif args.attack_mode == "c":
            # Mode c intentionally skips state coverage.
            pass
        else:
            raise NotImplementedError

        # Build the base mutation operators (e.g., blur, rotation, noise, etc.).
        mutation = get_img_mutations()

        # Dynamically choose the mutator-selector policy.
        # - ProbabilityImgMutations: adaptive/MCMC-like selector.
        # - RandomImgMutations: baseline random selector.
        if args.attack_mode == "a" or args.attack_mode == "c" or args.attack_mode == "d":
            from myLib.probability_img_mutations import ProbabilityImgMutations as ImgMutations
        elif args.attack_mode == "b":
            from myLib.probability_img_mutations import RandomImgMutations as ImgMutations
        else:
            raise NotImplementedError

        # Selector object also records per-operator statistics used for ranking.
        p_mutation = ImgMutations(mutation, args.seed)

        # Evaluate models on the original seed first.
        seed_org_result, seed_cps_result = predict_f(preprocessing(raw_seed_input))

        # Result object tracks full attack trace for this seed.
        result = SingleAttackResult(
            raw_seed_input,
            seed_label,
            idx,
            seed_org_result,
            seed_cps_result,
            save_dir,
        )
        start_time = time.time()

        # Early success: original and compressed model already disagree.
        if seed_org_result.label != seed_cps_result.label:
            logger("No need to search: org: {} vs cps: {}".format(seed_org_result.label, seed_cps_result.label))
            success_iter[idx] = 0
            result.update_results(None, seed_org_result, seed_cps_result, 0)

        else:
            # Initialize fitness for the selected mode.
            if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
                # Seed coverage enters corpus to establish starting state.
                _, _ = covered_states.update_function(np.hstack([seed_org_result.vec, seed_cps_result.vec]))
                from myLib.fitnessValue import StateFitnessValue as FitnessValue
                best_fitness_value = FitnessValue(False, 0)
            elif args.attack_mode == "c":
                from myLib.fitnessValue import DiffProbFitnessValue as FitnessValue
                best_fitness_value = FitnessValue(-1)
            else:
                raise NotImplementedError

            # Current local optimum / last accepted mutant.
            latest_img = np.copy(raw_seed_input)
            last_mutation_operator = None

            for iteration in range(1, args.maxit + 1):
                logger("Iteration： {}".format(iteration))

                # Enforce per-seed timeout strictly.
                if time.time() - start_time > args.timeout:
                    logger("Time Out")
                    break

                # Select mutation operator (adaptive or random per mode).
                m = p_mutation.choose_mutator(last_mutation_operator)
                m.total += 1  # Record how many times this operator was tried.
                logger("Mutator :{}".format(m.name))

                # Apply mutation on a copy to avoid in-place corruption.
                new_img = m.mut(np.copy(latest_img))

                # Evaluate both models on the new candidate.
                org_result, cps_result = predict_f(preprocessing(new_img))

                # Success condition: predicted labels are different.
                if org_result.label != cps_result.label:
                    logger("Found: org: {} vs cps: {}".format(org_result.label, cps_result.label))
                    success_iter[idx] = iteration
                    m.delta_bigger_than_zero += 1
                    result.update_results(new_img, org_result, cps_result, iteration)

                    # L2 distance from original seed quantifies perturbation size.
                    l2dist = np.linalg.norm((new_img - raw_seed_input).flatten(), ord=2)
                    l2distances[idx] = l2dist
                    break
                else:
                    # Probability difference is a basic discrepancy signal.
                    diff_prob = org_result.prob[0] - cps_result.prob[0]

                    if args.attack_mode == "a" or args.attack_mode == "b" or args.attack_mode == "d":
                        # Coverage vector combines both model outputs.
                        coverage = np.hstack([org_result.vec, cps_result.vec])

                        # Update corpus and compute whether a novel state is reached.
                        add_to_corpus, distance = covered_states.update_function(coverage)

                        # State-aware fitness couples novelty + probability discrepancy.
                        fitness_value = FitnessValue(add_to_corpus, diff_prob)
                    else:
                        # Mode c: only use probability difference.
                        fitness_value = FitnessValue(diff_prob)

                    # Accept candidate if it is no worse than the current best.
                    if fitness_value.better_than(best_fitness_value):
                        update_str = "update fitness value from {}".format(best_fitness_value)
                        best_fitness_value = fitness_value

                        if args.attack_mode == "d":
                            # Mode d gives distance-aware reward to mutation operator.
                            l2dist = np.linalg.norm((latest_img - new_img).flatten(), ord=2)
                            m.delta_bigger_than_zero += 1 / (l2dist ** -2)
                        else:
                            # Default: unit reward when fitness improves.
                            m.delta_bigger_than_zero += 1

                        # Move search center to the accepted candidate.
                        latest_img = np.copy(new_img)
                        last_mutation_operator = m
                        update_str += " to {}".format(best_fitness_value)
                        logger(update_str)

                    logger("Best " + str(best_fitness_value))

        # Persist this seed's result regardless of success/failure.
        result.save()

    # Output overall statistics after all seeds are processed.
    summary_attack_results(success_iter, logger, args.attack_mode)
