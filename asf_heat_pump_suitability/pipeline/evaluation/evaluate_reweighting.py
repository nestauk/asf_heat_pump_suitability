"""
Functions needed to calculate errors in the proportions of features per LSOA
"""

import polars as pl
import s3fs
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

import os
from typing import Tuple, Dict, List
from collections import defaultdict

from asf_heat_pump_suitability import PROJECT_DIR
from asf_heat_pump_suitability import config


def calculate_proportions(counts: dict) -> dict:
    """
    Calculate proportions for each key in a dictionary.
    Args:
            counts (dict): dictionary with counts for each key
    Returns:
            proportions (dict): dictionary with proportions for each key
    """
    total = sum(counts.values())
    proportions = {k: v / total for k, v in counts.items()}
    return proportions


def calculate_diffs(dict1: dict, dict2: dict) -> dict:
    """
    Calculate numerical differences between two dictionaries.
    Args:
            dict1 (dict): first dictionary
            dict2 (dict): second dictionary
    Returns:
            differences (dict): dictionary with differences for each key
    """
    keys = set(dict1) | set(dict2)
    differences = {k: dict1.get(k, 0) - dict2.get(k, 0) for k in keys}
    return differences


def get_error_metrics(
    sample_counts: dict,
    sample_proportions: dict,
    target_counts: dict,
    target_proportions: dict,
) -> dict:
    """
    Find the error metrics for one feature (e.g. Tenure) and one LSOA between the sample and target datasets.

    The RMSE and MAE outputs are in the range [0, 1], where 0 indicates very similar target and sample
    proportions, whereas a value close to 1 indicates they are very different.

    Args:
        sample_counts (dict): The counts for each category in the sample, e.g. {'private': 43, 'rental': 21}
        sample_proportions (dict): The proportions for each category in the sample, e.g. {'private': 0.672, 'rental': 0.328}
        target_counts (dict): The target counts for each category, e.g. {'private': 46, 'rental': 18}
        target_proportions (dict): The target proportions for each category, e.g. {'private': 0.719, 'rental': 0.281}
    Returns:
        dict: The evaluation metrics. This is as follows:
            {
            "counts_diff": The differences in the counts per category (sample minus target), e.g. {'private': -3, 'rental': 3}
            "proportions_diff": The differences in the proportions per category (sample minus target), e.g. {'private': -0.047, 'rental': 0.047}
            "rmse_no_missing_cats": The root mean squared error calculated on just the categories given in the sample proportions,
            "mae_no_missing_cats": The mean absolute error calculated on just the categories given in the sample proportions,
            "rmse_missing_cats": The root mean squared error calculated on all categories given in either the target or sample proportions,
            "mae_missing_cats": The mean absolute error calculated on all categories given in either the target or sample proportions,
            }

    """

    # Calculate differences between counts and proportions
    counts_diff = calculate_diffs(sample_counts, target_counts)
    proportions_diff = calculate_diffs(sample_proportions, target_proportions)

    # Ensure the keys are in the same order as in proportions
    ordered_target_dict_proportions = {
        key: target_proportions.get(key, 0) for key in sample_proportions
    }

    # Convert to a list
    ordered_list_target_dict_proportions = list(
        ordered_target_dict_proportions.values()
    )

    # Calculate RMSE and MAE without missing categories
    rmse_without_missing_categories = root_mean_squared_error(
        ordered_list_target_dict_proportions, list(sample_proportions.values())
    )
    mae_without_missing_categories = mean_absolute_error(
        ordered_list_target_dict_proportions, list(sample_proportions.values())
    )

    # Create a copy of proportions and add a zero for the missing category
    proportions_with_zero = sample_proportions.copy()

    # Find the union of the keys in both datasets
    all_keys = set(sample_proportions.keys()).union(set(target_proportions.keys()))

    # Ensure that both dictionaries contain all keys
    proportions_with_zero = {key: sample_proportions.get(key, 0) for key in all_keys}
    target_dict_proportions_with_zero = {
        key: target_proportions.get(key, 0) for key in all_keys
    }

    # Order target_dict_proportions based on proportions_with_zero
    ordered_target_dict_proportions_with_zero = {
        key: target_dict_proportions_with_zero[key] for key in proportions_with_zero
    }

    # Convert to a list
    ordered_list_target_dict_proportions_with_zero = list(
        ordered_target_dict_proportions_with_zero.values()
    )
    list_proportions_with_zero = list(proportions_with_zero.values())

    # Calculate RMSE and MAE with missing categories
    rmse_with_missing_categories = root_mean_squared_error(
        ordered_list_target_dict_proportions_with_zero, list_proportions_with_zero
    )
    mae_with_missing_categories = mean_absolute_error(
        ordered_list_target_dict_proportions_with_zero, list_proportions_with_zero
    )

    return {
        "counts_diff": counts_diff,
        "proportions_diff": proportions_diff,
        "rmse_no_missing_cats": rmse_without_missing_categories,
        "mae_no_missing_cats": mae_without_missing_categories,
        "rmse_missing_cats": rmse_with_missing_categories,
        "mae_missing_cats": mae_with_missing_categories,
    }


def get_error_reduction(
    before_proportions: dict,
    after_proportions: dict,
    target_proportions: dict,
) -> np.float64:
    """
    Find the average relative error reduction for one feature (e.g. Tenure) and one LSOA before and after reweighting relative to before reweighting.

    Args:
        before_proportions (dict): The proportions of each category in the sample data before weighting, e.g. {'private': 0.672, 'rental': 0.328}
        after_proportions (dict): The proportions of each category in the sample data after weighting.
        target_proportions (dict): The proportions of each category in the target data.

    Returns:
        np.float64: The average relative error reduction before and after reweighting across each category for a feature in an LSOA.
            This value is between -inf and 1

        Interpreting the output:
            - A negative value means weighting made the errors worse.
            - A positive value means weighting improved the results.
            - A very negative value (-inf) means weighting was much worse than no weighting.
            - A value close to 1 means weighting made the results much better.
            - A value of zero mean the errors with or without weighting were the same.
    """

    error_reduction_list = []
    for key in before_proportions:
        before_diff = np.abs(
            before_proportions.get(key, 0) - target_proportions.get(key, 0)
        )
        after_diff = np.abs(
            after_proportions.get(key, 0) - target_proportions.get(key, 0)
        )
        error_reduction = before_diff - after_diff
        if before_diff == 0:
            before_diff = 0.0000001  # To avoid division by 0
        # The percentage reduction in error when switching between before and after weighting
        error_reduction_list.append(error_reduction / before_diff)
    if error_reduction_list:
        return np.mean(error_reduction_list)
    else:
        return None


def process_single_lsoa(
    sample: pl.DataFrame,
    target: pl.DataFrame,
    test_lsoa: str,
    lsoa: str,
    feature_name: str = "tenure",
) -> Tuple[Dict, Dict, Dict, float, float]:
    """
    Process a single LSOA: calculate counts, differences, RMSE, and MAE.

    Args:
            sample (pl.DataFrame): The sample dataframe.
            target (pl.DataFrame): The target dataframe.
            test_lsoa (str): The LSOA to process.
            lsoa (str): literally the string "lsoa"
            feature_name (str): The name of the target feature.

    Returns:
            Tuple[Dict, Dict, Dict, float, float]: A tuple containing the counts, differences, RMSE, and MAE.
    """
    subset = sample.filter(pl.col(lsoa) == test_lsoa)
    target_subset = target.filter(pl.col(lsoa) == test_lsoa)
    counts_list = subset[feature_name].value_counts().to_dict()
    counts = dict(zip(counts_list[feature_name], counts_list["count"]))

    proportions = calculate_proportions(counts)
    list_proportions = list(proportions.values())

    target_dict = target_subset.to_dicts()[0]
    del target_dict[lsoa]
    target_dict_proportions = calculate_proportions(target_dict)

    error_metrics = get_error_metrics(
        counts, proportions, target_dict, target_dict_proportions
    )

    return (
        counts,
        error_metrics["counts_diff"],
        error_metrics["proportions_diff"],
        error_metrics["rmse_no_missing_cats"],
        error_metrics["mae_no_missing_cats"],
        error_metrics["rmse_missing_cats"],
        error_metrics["mae_missing_cats"],
    )


def process_all_lsoas(
    sample: pd.DataFrame,
    target: pd.DataFrame,
    test_lsoas: List[str],
    feature_name: str = "tenure",
) -> Tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, float]],
    Dict[str, int],
]:
    """
    Process all LSOAs and return dictionaries of RMSE, MAE, counts difference, proportions difference, and total counts.

    Args:
            sample (DataFrame): The EPC sample DataFrame.
            tenure (DataFrame): The tenure DataFrame.
            test_lsoas (list): A list of LSOAs to test.
            target_feature_name (str): The name of the target feature.

    Returns:
            rmse_all_without_missing_categories (dict): A dictionary with RMSE values for each LSOA without considering missing categories.
            mae_all_without_missing_categories (dict): A dictionary with MAE values for each LSOA without considering missing categories.
            rmse_all_with_missing_categories (dict): A dictionary with RMSE values for each LSOA with missing categories considered.
            mae_all_with_missing_categories (dict): A dictionary with MAE values for each LSOA with missing categories considered.
            counts_diff_all (dict): A dictionary with counts difference for each LSOA.
            proportions_diff_all (dict): A dictionary with proportions difference for each LSOA.
            total_counts (dict): A dictionary with total counts for each tenure type.
    """
    rmse_all_without_missing_categories = {}
    mae_all_without_missing_categories = {}
    rmse_all_with_missing_categories = {}
    mae_all_with_missing_categories = {}
    counts_diff_all = {}
    proportions_diff_all = {}
    total_counts = defaultdict(int)

    for test_lsoa in test_lsoas:
        (
            counts,
            counts_diff,
            proportions_diff,
            rmse_without_missing_categories,
            mae_without_missing_categories,
            rmse_with_missing_categories,
            mae_with_missing_categories,
        ) = process_single_lsoa(sample, target, test_lsoa, "lsoa", feature_name)
        rmse_all_without_missing_categories[test_lsoa] = rmse_without_missing_categories
        mae_all_without_missing_categories[test_lsoa] = mae_without_missing_categories
        rmse_all_with_missing_categories[test_lsoa] = rmse_with_missing_categories
        mae_all_with_missing_categories[test_lsoa] = mae_with_missing_categories
        counts_diff_all[test_lsoa] = counts_diff
        proportions_diff_all[test_lsoa] = proportions_diff
        for k, v in counts.items():
            total_counts[k] += v

    return (
        rmse_all_without_missing_categories,
        mae_all_without_missing_categories,
        rmse_all_with_missing_categories,
        mae_all_with_missing_categories,
        counts_diff_all,
        proportions_diff_all,
        total_counts,
    )


def create_error_dataframe_and_save(
    rmse_all_without_missing_categories: dict,
    mae_all_without_missing_categories: dict,
    rmse_all_with_missing_categories: dict,
    mae_all_with_missing_categories: dict,
    filename: str,
):
    """
    Create a DataFrame from RMSE and MAE dictionaries and save it to a CSV file.

    Args:
            rmse_all_without_missing_categories (dict): A dictionary with RMSE values for each LSOA without considering missing categories.
            mae_all_without_missing_categories (dict): A dictionary with MAE values for each LSOA without considering missing categories.
            rmse_all_with_missing_categories (dict): A dictionary with RMSE values for each LSOA with missing categories considered.
            mae_all_with_missing_categories (dict): A dictionary with MAE values for each LSOA with missing categories considered.
            filename (str): The name of the CSV file to save.
    """
    df_LSOA_errors = pl.DataFrame(
        {
            "LSOA": list(rmse_all_without_missing_categories.keys()),
            "RMSE without missing categories": list(
                rmse_all_without_missing_categories.values()
            ),
            "MAE without missing categories": list(
                mae_all_without_missing_categories.values()
            ),
            "RMSE with missing categories": list(
                rmse_all_with_missing_categories.values()
            ),
            "MAE with missing categories": list(
                mae_all_with_missing_categories.values()
            ),
        }
    )
    # Check if the directory exists, if not, create it
    path_filename = f"{PROJECT_DIR}/outputs/preweight_error_data/{filename}"
    directory = os.path.dirname(path_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df_LSOA_errors.write_csv(path_filename)


def create_dataframe_and_melt(dictionary, column_name, target_feature_name, lsoa):
    """
    Create a DataFrame from a dictionary and melt it.

    Args:
            dictionary (dict): The dictionary to convert to a DataFrame.
            column_name (str): The name of the value column after melting.
            target_feature_name (str): The name of the target feature.
            lsoa (str): literally just lsoa

    Returns:
            pl.DataFrame: The melted DataFrame.
    """
    df = pl.DataFrame(dictionary).transpose(include_header=True)
    df = df.rename({"column": lsoa})
    df_melted = df.melt(
        id_vars=lsoa, variable_name=target_feature_name, value_name=column_name
    )
    return df_melted


def extract_values(df, column_name, target_feature_name, lsoa):
    """
    Extract values from a DataFrame into a list of dictionaries.

    Args:
            df (pl.DataFrame): The DataFrame to extract values from.
            column_name (str): The column to extract values from.
            target_feature_name (str): The name of the target feature.
            lsoa (str): literally just lsoa

    Returns:
            list: A list of dictionaries with the extracted values.
    """
    melted_rows = []
    for row in df.rows(named=True):
        for key, value in row[column_name].items():
            melted_rows.append(
                {lsoa: row[lsoa], target_feature_name: key, column_name: value}
            )
    return melted_rows


def create_boxplot(df, y, title, filename, target_feature_name, order_preference):
    """
    Create a boxplot from a DataFrame and save it to a file.

    Args:
            df (pd.DataFrame): The DataFrame to create a boxplot from.
            y (str): The column to use as the y-axis.
            title (str): The title of the plot.
            filename (str): The name of the file to save the plot to.
            target_feature_name (str): The name of the target feature to be displayed on the x-axis.
            order_preference (List[str]): A list of categories in the order of preference for plotting.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=target_feature_name, y=y, data=df, ax=ax, order=order_preference)
    for i, target_feature in enumerate(order_preference):
        ax.text(
            i,
            ax.get_ylim()[1] + 0.01,
            f"$n_{{properties}}$ = {total_counts[target_feature]}",
            ha="center",
        )
    ax.set_title(title, y=1.05)
    ax.set_ylabel(f"$\Delta_{{{y.split(' ')[1]}}}$", labelpad=15, fontsize=14)
    plt.tight_layout()
    path_filename = f"{PROJECT_DIR}/outputs/figures/preweight_errors/{filename}"
    directory = os.path.dirname(path_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(path_filename)
    plt.show()


def create_difference_plots(
    differences_all: dict,
    title: str,
    filename: str,
    label: str,
    target_feature_name: str,
    order_preference: List[str],
) -> None:
    """
    Create a grid of bar plots for each LSOA showing the differences.

    Args:
            differences_all (dict): A dictionary of dictionaries where the outer keys are LSOAs and the inner keys are tenures.
            title (str): The title for the plot.
            filename (str): The name of the file to save the plot to.
            label (str): The label for the y-axis (e.g., "$\Delta_{count}$" or "$\Delta_{proportion}$"
            target_feature_name (str): The name of the target feature to be displayed on the x-axis.
            order_preference (List[str]): A list of categories in the order of preference for plotting.

    Returns:
            None
    """
    # List of LSOAs
    lsoas = list(differences_all.keys())

    # Number of LSOAs
    n_lsoas = len(lsoas)

    # Calculate the number of rows and columns for the grid
    n_cols = 3
    n_rows = math.ceil(n_lsoas / n_cols)

    # Create a figure with a subplot for each LSOA
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    # Find the global minimum and maximum differences
    global_min = min(min(differences_all[lsoa].values()) for lsoa in lsoas)
    global_max = max(max(differences_all[lsoa].values()) for lsoa in lsoas)
    global_min_buffer = global_min + 0.1 * global_min
    global_max_buffer = global_max + 0.1 * global_max

    # Get the color palette used by seaborn's boxplot
    palette = sns.color_palette()

    # Loop over the LSOAs
    for i, lsoa in enumerate(lsoas):
        # Create a DataFrame for the current LSOA
        df = pd.DataFrame(
            {
                target_feature_name: list(differences_all[lsoa].keys()),
                "Difference": list(
                    map(float, differences_all[lsoa].values())
                ),  # Convert to float
            }
        )

        # Create a bar chart in the current subplot
        df.plot(
            x=target_feature_name,
            y="Difference",
            kind="bar",
            ax=axs[i],
            color=[palette[order_preference.index(tenure)] for tenure in df["tenure"]],
            legend=False,
        )

        # Set the title and labels
        axs[i].set_title(f"$\\Delta_{{{label}}}$ for {lsoa}")
        axs[i].set_ylabel(f"$\\Delta_{{{label}}}$")
        axs[i].set_xlabel(target_feature_name)
        axs[i].set_xticklabels(df[target_feature_name], rotation=15)

        # Set the y-axis limits
        axs[i].set_ylim(global_min_buffer, global_max_buffer)

        # Add a gridline at 0
        axs[i].axhline(0, color="black", linewidth=0.5)

    # Remove unused subplots
    for i in range(n_lsoas, n_rows * n_cols):
        fig.delaxes(axs[i])

    # Show the plot
    fig.suptitle(title, y=0.98)

    plt.tight_layout()
    plt.show()
    path_filename = f"{PROJECT_DIR}/outputs/figures/preweight_errors/{filename}"
    directory = os.path.dirname(path_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(path_filename)
