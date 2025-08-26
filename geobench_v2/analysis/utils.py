# Copyright (c) 2025 GeoBenchV2. All rights reserved.
# Licensed under the Apache License 2.0.

"""Utility functions for GeoBenchV2 result analysis."""

import os

import pandas as pd
from omegaconf import OmegaConf


def collect_overview_df(exp_dir: str, select_criteria: str) -> pd.DataFrame:
    """Collect overview metric results from Lightning CSV Logger into a dataframe.

    Args:
        exp_dir: path to experiment directory
        select_criteria: criteria to select the best model, can either
            be "loss" or "metric". in case of "metric", for classification
            tasks, the metric is "ACC" and for regression tasks, the
            metric is "MSE", and for segmentation tasks, the metric is
            "IoU".

    Returns:
        a pandas dataframe with overview of results
    """
    assert select_criteria in ["loss", "metric"], (
        "select_criteria must be either 'loss' or 'metric'"
    )

    # Initialize a list to hold the combined data
    results = []

    # Iterate over each experiment subdirectory
    for exp_name in os.listdir(exp_dir):
        exp_path = os.path.join(exp_dir, exp_name)
        if os.path.isdir(exp_path):
            # Paths to config and metrics files
            config_path = os.path.join(exp_path, "config.yaml")
            metrics_path = os.path.join(
                exp_path, "lightning_logs", "version_0", "metrics.csv"
            )

            if os.path.exists(config_path) and os.path.exists(metrics_path):
                config = OmegaConf.load(config_path)
                metrics_df = pd.read_csv(metrics_path)
                if "epoch" not in metrics_df.columns:
                    print(
                        f"'epoch' column not found in metrics.csv for {exp_name}, looks like a failed run run."
                    )
                    continue

                task = config.dataset_config.task

                if task == "classification":
                    metric_name = {"loss": "val_loss_epoch", "metric": "val_acc1_epoch"}
                elif task == "segmentation":
                    metric_name = {"loss": "val_loss_epoch", "metric": "val_miou_epoch"}
                else:
                    print(f"Task {task} not supported")

                if metric_name[select_criteria] not in metrics_df.columns:
                    print(
                        f"{metric_name[select_criteria]} not found in metrics.csv for {exp_name}, skipping"
                    )
                    continue
                if select_criteria == "loss":
                    best_val_idx = metrics_df[metric_name[select_criteria]].idxmin()
                else:
                    best_val_idx = metrics_df[metric_name[select_criteria]].idxmax()

                best_val_row = metrics_df.loc[best_val_idx]

                # CSV logger tracks train metrics in the row after validation results
                if best_val_idx + 1 < len(metrics_df):
                    next_row = metrics_df.loc[best_val_idx + 1]
                    train_metrics_cols = [
                        col
                        for col in metrics_df.columns
                        if col.startswith("train") and col != "train_loss"
                    ]
                    train_metrics = (
                        next_row[train_metrics_cols]
                        if not next_row[train_metrics_cols].isnull().all()
                        else pd.Series(dtype=float)
                    )
                else:
                    train_metrics = pd.Series(dtype=float)

                # Combine the metrics
                combined_metrics = best_val_row.copy()
                for col in train_metrics.index:
                    combined_metrics[col] = train_metrics[col]

                # Combine config and metrics into the result
                result = {"experiment": exp_name.split("_")[0]}
                result["task"] = task
                result["dataset"] = config.args.dataset
                result["exp_dir"] = exp_path

                # exclude step metrics
                combined_metrics = combined_metrics[
                    ~combined_metrics.index.str.contains("step")
                ]
                combined_metrics = combined_metrics[
                    ~combined_metrics.index.str.contains("test_")
                ]

                combined_metrics = combined_metrics[
                    ~combined_metrics.index.str.startswith("lr-")
                ]

                # rename the actual task specific to a "metric" to have a common name
                # across tasks for train and val
                if task == "classification":
                    combined_metrics["val_metric"] = combined_metrics.pop(
                        "val_acc1_epoch"
                    )
                    combined_metrics["train_metric"] = combined_metrics.pop(
                        "train_acc1_epoch"
                    )
                    combined_metrics.drop("val_acc5_epoch", inplace=True)
                    combined_metrics.drop("train_acc5_epoch", inplace=True)

                elif task == "segmentation":
                    combined_metrics["val_metric"] = combined_metrics.pop(
                        "val_miou_epoch"
                    )
                    combined_metrics["train_metric"] = combined_metrics.pop(
                        "train_miou_epoch"
                    )
                    combined_metrics.drop("val_acc_epoch", inplace=True)
                    combined_metrics.drop("train_acc_epoch", inplace=True)

                for col in combined_metrics.index:
                    result[col] = combined_metrics[col]
                # Append to the results list
                results.append(result)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df


def find_best_model(results_df, best_metric):
    """Finds the best model based on the specified metric(s).

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.
        best_metric: metric name basd on which to select the best experiment

    Returns:
        pd.Series: The row corresponding to the best model.
    """
    if best_metric.endswith("loss") or "loss" in best_metric.lower():
        # For loss metrics, lower is better
        idx = results_df.groupby("experiment")[best_metric].idxmin()
    else:
        # For other metrics, higher is better
        idx = results_df.groupby("experiment")[best_metric].idxmax()

    best_results_df = results_df.loc[idx].reset_index(drop=True)
    return best_results_df


exp_dir = "/mnt/rg_climate_benchmark/results/nils/exps"

results_df = collect_overview_df(exp_dir, "metric")
results_df
