"""
This script performs an error analysis on the tenure data by LSOA (Lower Layer Super Output Area).

First, it loads the sample and target data. Then, it processes all LSOAs to get error metrics and differences in counts and proportions. These metrics are saved to a CSV file.

Next, it creates and melts dataframes for counts and proportions differences, extracts values from these melted dataframes, and converts them to pandas dataframes.

Finally, it creates boxplots and difference plots for counts and proportions differences. These plots provide a visual representation of the differences in counts and proportions by tenure for each LSOA.

This script is intended to be run as a standalone script.
"""

import polars as pl
import s3fs
from asf_heat_pump_suitability import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from typing import Tuple, Dict, List
from collections import defaultdict
from asf_heat_pump_suitability import PROJECT_DIR
import os

sample_path = "s3://asf-heat-pump-suitability/outputs/epc_sample_lsoa.parquet"
target_path = config["data_source"]["EW_housing_characteristics_census"]
DELTA_COUNT = "Delta count"
DELTA_PROPORTION = "Delta proportion"
ORDER = ["owner-occupied", "rental (private)", "rental (social)"]
LSOA = "lsoa"
TENURE = "tenure"
test_lsoas = ["W01000328", "E01033942", "E01012376", "W01000527", "E01002058"]
number_of_lsoas = len(test_lsoas)


def load_sample(sample_path):
    """
    Load the EPC sample from a parquet file and preprocess it.

    Args:
        sample_path (str): The path to the parquet file containing the EPC sample.

    Returns:
        sample (pl.DataFrame): The preprocessed EPC sample, with irrelevant rows filtered out and columns renamed.
    """
    sample = pl.read_parquet(sample_path)
    sample = sample.filter(~pl.col("TENURE").is_in(["", "unknown"]))
    sample = sample.select(pl.col(["lsoa", "UPRN", "TENURE"])).rename(
        {"TENURE": "tenure", "UPRN": "id"}
    )
    return sample


def load_target(target_path):
    """
    Load the target dataset from the census data.

    Args:
        target_path (str): The path to the Excel file containing the census data.

    Returns:
        tenure (pl.DataFrame): The preprocessed census data, with irrelevant rows and columns removed, and columns renamed.
    """
    fs = s3fs.S3FileSystem()
    with fs.open(target_path, mode="rb") as f:
        content = f.read()
    tenure = pl.read_excel(content, sheet_name="3c", engine="calamine")
    tenure = tenure.rename(tenure[2].to_dicts().pop()).slice(3)
    tenure = tenure.rename(
        {
            "Area Code": "lsoa",
            "Owned or shared ownership": "owner-occupied",
            "Social Rented": "rental (social)",
            "Private Rented or lives rent free": "rental (private)",
        }
    )
    censored_vals = 0
    int_cols = ["owner-occupied", "rental (private)", "rental (social)"]
    tenure = tenure.with_columns(
        [pl.col(int_cols).str.replace("c", f"{censored_vals}").cast(pl.Int64)]
    ).select(pl.col(["lsoa", "owner-occupied", "rental (private)", "rental (social)"]))
    return tenure


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
    Calculate differences between two dictionaries.
    Args:
        dict1 (dict): first dictionary
        dict2 (dict): second dictionary
    Returns:
        differences (dict): dictionary with differences for each key
    """
    keys = set(dict1) | set(dict2)
    differences = {k: dict1.get(k, 0) - dict2.get(k, 0) for k in keys}
    return differences


def calculate_rmse(proportions_diff: dict) -> float:
    """
    Calculate Root Mean Square Error (RMSE) from a dictionary of differences.
    Args:
        proportions_diff (dict): dictionary with differences for each key
    Returns:
        float: RMSE
    """
    squares = [x**2 for x in proportions_diff.values()]
    mean_square = np.mean(squares)
    rmse = np.sqrt(mean_square)
    return rmse


def calculate_mae(proportions_diff: dict) -> float:
    """
    Calculate Mean Absolute Error (MAE) from a dictionary of differences.
    Args:
        proportions_diff (dict): dictionary with differences for each key
    Returns:
        float: MAE
    """
    absolute_values = [abs(x) for x in proportions_diff.values()]
    mae = np.mean(absolute_values)
    return mae


def process_single_lsoa(
    sample: pl.DataFrame, tenure: pl.DataFrame, test_lsoa: str
) -> Tuple[Dict, Dict, Dict, float, float]:
    """
    Process a single LSOA: calculate counts, differences, RMSE, and MAE.

    Args:
        sample (pl.DataFrame): The sample dataframe.
        tenure (pl.DataFrame): The tenure dataframe.
        test_lsoa (str): The LSOA to process.

    Returns:
        Tuple[Dict, Dict, Dict, float, float]: A tuple containing the counts, differences, RMSE, and MAE.
    """
    subset = sample.filter(pl.col(LSOA) == test_lsoa)
    tenure_subset = tenure.filter(pl.col(LSOA) == test_lsoa)

    counts_subset_epc_sample = subset.group_by(TENURE).agg(
        [pl.col(TENURE).count().alias("count")]
    )
    counts = dict(
        zip(counts_subset_epc_sample[TENURE], counts_subset_epc_sample["count"])
    )

    proportions = calculate_proportions(counts)

    tenure_dict = tenure_subset.to_dicts()[0]
    del tenure_dict[LSOA]
    tenure_dict_proportions = calculate_proportions(tenure_dict)

    counts_diff = calculate_diffs(counts, tenure_dict)
    proportions_diff = calculate_diffs(proportions, tenure_dict_proportions)

    rmse = calculate_rmse(proportions_diff)
    mae = calculate_mae(proportions_diff)
    return counts, counts_diff, proportions_diff, rmse, mae


def process_all_lsoas(
    sample: pd.DataFrame, tenure: pd.DataFrame, test_lsoas: List[str]
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

    Returns:
        rmse_all (dict): A dictionary with RMSE values for each LSOA.
        mae_all (dict): A dictionary with MAE values for each LSOA.
        counts_diff_all (dict): A dictionary with counts difference for each LSOA.
        proportions_diff_all (dict): A dictionary with proportions difference for each LSOA.
        total_counts (dict): A dictionary with total counts for each tenure type.
    """
    rmse_all = {}
    mae_all = {}
    counts_diff_all = {}
    proportions_diff_all = {}
    total_counts = defaultdict(int)

    for test_lsoa in test_lsoas:
        counts, counts_diff, proportions_diff, rmse, mae = process_single_lsoa(
            sample, tenure, test_lsoa
        )
        rmse_all[test_lsoa] = rmse
        mae_all[test_lsoa] = mae
        counts_diff_all[test_lsoa] = counts_diff
        proportions_diff_all[test_lsoa] = proportions_diff
        for k, v in counts.items():
            total_counts[k] += v

    return rmse_all, mae_all, counts_diff_all, proportions_diff_all, total_counts


def create_error_dataframe_and_save(rmse_all: dict, mae_all: dict, filename: str):
    """
    Create a DataFrame from RMSE and MAE dictionaries and save it to a CSV file.

    Args:
        rmse_all (dict): A dictionary with RMSE values for each LSOA.
        mae_all (dict): A dictionary with MAE values for each LSOA.
        filename (str): The name of the CSV file to save.
    """
    df_LSOA_errors = pl.DataFrame(
        {
            "LSOA": list(rmse_all.keys()),
            "RMSE": list(rmse_all.values()),
            "MAE": list(mae_all.values()),
        }
    )
    # Check if the directory exists, if not, create it
    path_filename = f"{PROJECT_DIR}/outputs/preweight_error_data/{filename}"
    directory = os.path.dirname(path_filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df_LSOA_errors.write_csv(path_filename)


def create_dataframe_and_melt(dictionary, column_name):
    """
    Create a DataFrame from a dictionary and melt it.

    Args:
        dictionary (dict): The dictionary to convert to a DataFrame.
        column_name (str): The name of the value column after melting.

    Returns:
        pl.DataFrame: The melted DataFrame.
    """
    df = pl.DataFrame(dictionary).transpose(include_header=True)
    df = df.rename({"column": LSOA})
    df_melted = df.melt(id_vars=LSOA, variable_name=TENURE, value_name=column_name)
    return df_melted


def extract_values(df, column_name):
    """
    Extract values from a DataFrame into a list of dictionaries.

    Args:
        df (pl.DataFrame): The DataFrame to extract values from.
        column_name (str): The column to extract values from.

    Returns:
        list: A list of dictionaries with the extracted values.
    """
    melted_rows = []
    for row in df.rows(named=True):
        for key, value in row[column_name].items():
            melted_rows.append({LSOA: row[LSOA], TENURE: key, column_name: value})
    return melted_rows


def create_boxplot(df, y, title, filename):
    """
    Create a boxplot from a DataFrame and save it to a file.

    Args:
        df (pd.DataFrame): The DataFrame to create a boxplot from.
        y (str): The column to use as the y-axis.
        title (str): The title of the plot.
        filename (str): The name of the file to save the plot to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=TENURE, y=y, data=df, ax=ax, order=ORDER)
    for i, tenure in enumerate(ORDER):
        ax.text(
            i,
            ax.get_ylim()[1] + 0.01,
            f"$n_{{properties}}$ = {total_counts[tenure]}",
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
    differences_all: dict, title: str, filename: str, label: str
) -> None:
    """
    Create a grid of bar plots for each LSOA showing the differences.

    Args:
        differences_all (dict): A dictionary of dictionaries where the outer keys are LSOAs and the inner keys are tenures.
        title (str): The title for the plot.
        filename (str): The name of the file to save the plot to.
        label (str): The label for the y-axis (e.g., "$\Delta_{count}$" or "$\Delta_{proportion}$"

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
                "Tenure": list(differences_all[lsoa].keys()),
                "Difference": list(
                    map(float, differences_all[lsoa].values())
                ),  # Convert to float
            }
        )

        # Create a bar chart in the current subplot
        df.plot(
            x="Tenure",
            y="Difference",
            kind="bar",
            ax=axs[i],
            color=[palette[ORDER.index(tenure)] for tenure in df["Tenure"]],
            legend=False,
        )

        # Set the title and labels
        axs[i].set_title(f"$\\Delta_{{{label}}}$ for {lsoa}")
        axs[i].set_ylabel(f"$\\Delta_{{{label}}}$")
        axs[i].set_xlabel("Tenure")
        axs[i].set_xticklabels(df["Tenure"], rotation=15)

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


if __name__ == "__main__":
    # Load the sample and target data
    sample = load_sample(sample_path)
    tenure = load_target(target_path)

    # Process all LSOAs and get error metrics and differences
    rmse_all, mae_all, counts_diff_all, proportions_diff_all, total_counts = (
        process_all_lsoas(sample, tenure, test_lsoas)
    )

    # Save the error metrics to a CSV file
    create_error_dataframe_and_save(rmse_all, mae_all, "tenure_lsoa_errors.csv")

    # Create and melt dataframes for counts and proportions differences
    counts_df_melted = create_dataframe_and_melt(counts_diff_all, DELTA_COUNT)
    proportions_df_melted = create_dataframe_and_melt(
        proportions_diff_all, DELTA_PROPORTION
    )

    # Extract values from the melted dataframes
    melted_rows = extract_values(counts_df_melted, DELTA_COUNT)
    counts_df_extracted = pl.DataFrame(melted_rows)
    proportion_melted_rows = extract_values(proportions_df_melted, DELTA_PROPORTION)
    proportions_df_extracted = pl.DataFrame(proportion_melted_rows)

    # Convert to pandas dataframes
    counts_df_extracted_pd = counts_df_extracted.to_pandas()
    proportions_df_extracted_pd = proportions_df_extracted.to_pandas()

    # Create boxplots for counts and proportions differences
    create_boxplot(
        counts_df_extracted_pd,
        DELTA_COUNT,
        f"Boxplot of pre-weighted $\\Delta_{{count}}$ by Tenure for $n_{{LSOA}}={number_of_lsoas}$",
        "delta_count_boxplot.png",
    )
    create_boxplot(
        proportions_df_extracted_pd,
        DELTA_PROPORTION,
        f"Boxplot of pre-weighted $\\Delta_{{proportion}}$ by Tenure for $n_{{LSOA}}={number_of_lsoas}$",
        "delta_proportion_boxplot.png",
    )

    # Create difference plots for counts and proportions differences
    create_difference_plots(
        counts_diff_all,
        r"Pre-weighted $\Delta_{count}(n_{\mathrm{EPC\ properties}} - n_{\mathrm{census\ properties}})$",
        "preweighted_delta_count_LSOAs.png",
        "count",
    )
    create_difference_plots(
        proportions_diff_all,
        r"Pre-weighted $\Delta_{proportion}$",
        "preweighted_delta_proportion_LSOAs.png",
        "proportion",
    )
