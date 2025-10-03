# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions to analyse results from GeoBenchV2."""

import math
import random
from itertools import combinations, product

import pandas as pd
from scipy import stats

REPEATED_SEEDS_DEFAULT = 5


def discriminativity(scores_i, scores_j, *, tie_half=True, eps=1e-12):
    """Returns Discriminativity d_ij^l = 1 − H₂[p(A_i > A_j)] (binary entropy in bits).

    Args:
        scores_i: scores for method i
        scores_j: scores for method j
        tie_half: bool, describes whether to split ties
        eps: epsilon value

    Returns:
        1.0 - h2
    """
    total = len(scores_i) * len(scores_j)
    better = ties = 0
    for s_i, s_j in product(scores_i, scores_j):
        if s_i > s_j:
            better += 1
        elif s_i == s_j:
            ties += 1

    if tie_half:
        better += 0.5 * ties  # split ties
    p = max(min(better / total, 1 - eps), eps)

    h2 = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return 1.0 - h2


def dataset_discriminativity(score_lists, tie_half=True, eps=1e-12):
    """Average discriminativity over every unordered pair of m algorithms.

    D_l = (2 / (m·(m−1))) · Σ_{i<j} d_ij^l

    Args:
        score_lists : list[list[float]]
            score_lists[k] holds the seed-level scores for algorithm k.
        tie_half: bool, describes whether to split ties
        eps: epsilon value

    Returns:
        float in [0, 1]
    """
    pairs = combinations(range(len(score_lists)), 2)
    vals = [
        discriminativity(score_lists[i], score_lists[j], tie_half=tie_half, eps=eps)
        for i, j in pairs
    ]

    return sum(vals) / len(vals) if vals else 0.0


def bootstrap_dataset_discriminativity(
    score_lists, n_iter=100, ci=0.95, tie_half=True, eps=1e-12, random_state=None
):
    """Stratified bootstrapped discriminativity.

    Resample seeds **within each algorithm** with
    replacement, recompute D_l, repeat n_iter times.

    Args:
        score_lists : list[list[float]]
            score_lists[k] holds the seed-level scores for algorithm k.
        n_iter: number of iterations for bootstrapping
        ci: confidence interval
        tie_half: bool, describes whether to split ties
        eps: epsilon value
        random_state: if not None, fixed random state to be used

    Returns:
        dict with mean, (lower, upper) CI, and all bootstrap samples.
    """
    rng = random.Random(random_state)
    boot = []
    for _ in range(n_iter):
        resampled = [rng.choices(scores, k=len(scores)) for scores in score_lists]
        boot.append(dataset_discriminativity(resampled, tie_half=tie_half, eps=eps))

    boot.sort()
    k = int((1 - ci) / 2 * n_iter)
    return {
        "mean": sum(boot) / n_iter,
        "ci": (boot[k], boot[-k - 1]),  # two-sided percentile CI
        "samples": boot,  # keep if you want the full distribution
    }


def compute_entropy_based_discriminativity(
    results: pd.DataFrame, num_repetitions: int = REPEATED_SEEDS_DEFAULT
):
    """Computes variance based discriminativity per dataset.

    Args:
        results: dataframe of extracted results
        num_repetitions: number of repetitions used in repeated stage

    Returns:
        pd.DataFrame with results
    """
    datasets = results["dataset"].tolist()
    datasets = sorted(set(datasets))
    list_of_dataset = []
    list_of_scores = []
    list_of_bootstrapped_mean = []
    list_of_bootstrapped_ci = []
    for dataset in datasets:
        results_for_dataset = results.loc[results["dataset"] == dataset].copy()
        counts = (
            results_for_dataset[["experiment_name", "Seed"]]
            .groupby("experiment_name")
            .count()
        )

        # only include experiments which have the required number of repetitions
        experiment_names = counts.loc[counts["Seed"] == num_repetitions].index.tolist()
        results_for_dataset = results_for_dataset.loc[
            results_for_dataset["experiment_name"].isin(experiment_names)
        ].copy()
        results_for_dataset = (
            results_for_dataset.groupby("experiment_name")["test metric"]
            .apply(list)
            .reset_index(name="metric")
        )
        metrics = results_for_dataset["metric"].tolist()

        # Compute discriminativity for dataset
        result = dataset_discriminativity(metrics, tie_half=True, eps=1e-12)
        bootstrapped_uncertainty = bootstrap_dataset_discriminativity(
            metrics, n_iter=100, ci=0.95, tie_half=True, eps=1e-12, random_state=None
        )
        list_of_dataset.append(dataset)
        list_of_scores.append(result)
        list_of_bootstrapped_mean.append(bootstrapped_uncertainty["mean"])
        list_of_bootstrapped_ci.append(bootstrapped_uncertainty["ci"])

    output = pd.DataFrame(
        {
            "dataset": list_of_dataset,
            "discriminativity": list_of_scores,
            "bootstrapped_mean": list_of_bootstrapped_mean,
            "bootstrapped_ci": list_of_bootstrapped_ci,
        }
    )
    return output


def compute_variance_based_discriminativity(
    results: pd.DataFrame, num_repetitions: int = REPEATED_SEEDS_DEFAULT
):
    """Computes variance based discriminativity per dataset.

    Args:
        results: dataframe of extracted results
        num_repetitions: number of repetitions used in repeated stage

    Returns:
        pd.DataFrame with results
    """
    datasets = results["dataset"].tolist()
    datasets = sorted(set(datasets))
    list_of_dataset = []
    list_of_stats = []
    list_of_p_values = []
    list_of_log_p_values = []
    for dataset in datasets:
        results_for_dataset = results.loc[results["dataset"] == dataset].copy()
        counts = (
            results_for_dataset[["experiment_name", "Seed"]]
            .groupby("experiment_name")
            .count()
        )

        # only include experiments which have the required number of repetitions
        experiment_names = counts.loc[counts["Seed"] == num_repetitions].index.tolist()
        results_for_dataset = results_for_dataset.loc[
            results_for_dataset["experiment_name"].isin(experiment_names)
        ].copy()
        results_for_dataset = (
            results_for_dataset.groupby("experiment_name")["test metric"]
            .apply(list)
            .reset_index(name="metric")
        )
        metrics = results_for_dataset["metric"].tolist()

        list_of_dataset.append(dataset)
        # Conduct the Kruskal-Wallis Test
        if len(experiment_names) == 1:
            list_of_stats.append("NA")
            list_of_p_values.append("NA")
            list_of_log_p_values.append("NA")

        else:
            result = stats.kruskal(*metrics)
            list_of_stats.append(result.statistic)
            list_of_p_values.append(result.pvalue)
            list_of_log_p_values.append(math.log(result.pvalue))

    output = pd.DataFrame(
        {
            "dataset": list_of_dataset,
            "p_value": list_of_p_values,
            "log_p_value": list_of_log_p_values,
            "statistic": list_of_stats,
        }
    )
    return output
