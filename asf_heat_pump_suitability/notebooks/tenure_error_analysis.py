import random
import polars as pl
import balance
import s3fs
from asf_heat_pump_suitability import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np
from typing import Tuple, Dict
from collections import defaultdict

# Import the EPC sample

sample_path = "s3://asf-heat-pump-suitability/outputs/epc_sample_lsoa.parquet"
sample = pl.read_parquet(sample_path)
sample["lsoa"].value_counts().sort(by="count").to_dicts()

# Prepare the EPC sample for the analysis

sample = sample.filter(~pl.col("TENURE").is_in(["", "unknown"]))


# Select relevant cols from sample
sample = sample.select(pl.col(["lsoa", "UPRN", "TENURE"])).rename(
    {"TENURE": "tenure", "UPRN": "id"}
)

print(sample.head())


# IMPORT THE VALIDATION DATASET (TENURE) FROM THE CENSUS DATA

# Load target dataset from census
target_path = config["data_source"]["EW_housing_characteristics_census"]
fs = s3fs.S3FileSystem()
with fs.open(target_path, mode="rb") as f:
    content = f.read()

# Load tenure data from census
tenure = pl.read_excel(content, sheet_name="3c", engine="calamine")

tenure = tenure.rename(tenure[2].to_dicts().pop())
tenure = tenure.slice(
    3,
)
tenure = tenure.rename(
    {
        "Area Code": "lsoa",
        "Owned or shared ownership": "owner-occupied",
        "Social Rented": "rental (social)",
        "Private Rented or lives rent free": "rental (private)",
    }
)
# Replace censored values ("c") with 0 value and convert cols to int
censored_vals = 0

int_cols = ["owner-occupied", "rental (private)", "rental (social)"]
tenure = tenure.with_columns(
    [pl.col(int_cols).str.replace("c", f"{censored_vals}").cast(pl.Int64)]
).select(pl.col(["lsoa", "owner-occupied", "rental (private)", "rental (social)"]))

print(tenure.head())

# Choose test LSOA
test_lsoas = ["W01000328", "E01033942", "E01012376", "W01000527", "E01002058"]
number_of_lsoas = len(test_lsoas)


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


def process_lsoa(
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


LSOA = "lsoa"
TENURE = "tenure"
total_counts = defaultdict(int)
rmse_all = {}
mae_all = {}
counts_diff_all = {}
proportions_diff_all = {}

# Iterate over all LSOAs
for test_lsoa in test_lsoas:
    counts, counts_diff, proportions_diff, rmse, mae = process_lsoa(
        sample, tenure, test_lsoa
    )
    rmse_all[test_lsoa] = rmse
    mae_all[test_lsoa] = mae
    counts_diff_all[test_lsoa] = counts_diff
    proportions_diff_all[test_lsoa] = proportions_diff
    for k, v in counts.items():
        total_counts[k] += v

print("RMSE for each LSOA:")
print(rmse_all)
print("MAE for each LSOA:")
print(mae_all)


# Step 1: Convert the dictionary to a Polars DataFrame and transpose
counts_df = pl.DataFrame(counts_diff_all).transpose(include_header=True)
counts_df = counts_df.rename({"column": "LSOA"})
proportions_df = pl.DataFrame(proportions_diff_all).transpose(include_header=True)
proportions_df = proportions_df.rename({"column": "LSOA"})

# Step 2: Melt the DataFrame
counts_df_melted = counts_df.melt(
    id_vars="LSOA", variable_name="Tenure", value_name="Delta Count"
)
proportions_df_melted = proportions_df.melt(
    id_vars="LSOA", variable_name="Tenure", value_name="Delta Proportion"
)


# Extract the numeric values from the Delta structs
def extract_values(df, column_name="Delta Count"):
    melted_rows = []
    for row in df.rows(named=True):
        for key, value in row[column_name].items():
            melted_rows.append({"LSOA": row["LSOA"], "Tenure": key, column_name: value})
    return melted_rows


# Apply the function to extract values
melted_rows = extract_values(counts_df_melted)
counts_df_extracted = pl.DataFrame(melted_rows)
# Apply the function to extract values
proportion_melted_rows = extract_values(proportions_df_melted, "Delta Proportion")
proportions_df_extracted = pl.DataFrame(proportion_melted_rows)

# Step 4: Convert the Polars DataFrame to a Pandas DataFrame for plotting
counts_df_extracted_pd = counts_df_extracted.to_pandas()
proportions_df_extracted_pd = proportions_df_extracted.to_pandas()

# Step 5: Create the boxplots
# Define the order of the hue levels
order = ["owner-occupied", "rental (private)", "rental (social)"]
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    x="Tenure", y="Delta Count", data=counts_df_extracted_pd, ax=ax, order=order
)
# Add the sample sizes to the plot
for i, tenure in enumerate(order):
    ax.text(
        i,
        ax.get_ylim()[1] + 7.0,
        f"$n_{{properties}}$ = {total_counts[tenure]}",
        ha="center",
    )
# Add a title to the plot
ax.set_title(
    f"Boxplot of pre-weighted $\\Delta_{{count}}$ by Tenure for $n_{{LSOA}}={number_of_lsoas}$",
    y=1.05,
)
ax.set_ylabel("$\Delta_{count}$", labelpad=15, fontsize=14)
plt.tight_layout()
# Save the plot as a PNG file
fig.savefig("delta_count_boxplot.png")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    x="Tenure",
    y="Delta Proportion",
    data=proportions_df_extracted_pd,
    ax=ax,
    order=order,
)
# Add the sample sizes to the plot
for i, tenure in enumerate(order):
    ax.text(
        i,
        ax.get_ylim()[1] + 0.01,
        f"$n_{{properties}}$ = {total_counts[tenure]}",
        ha="center",
    )
# Add a title to the plot
ax.set_title(
    f"Boxplot of pre-weighted $\\Delta_{{proportion}}$ by Tenure for $n_{{LSOA}}={number_of_lsoas}$",
    y=1.05,
)
ax.set_ylabel("$\Delta_{proportion}$", labelpad=15, fontsize=14)
plt.tight_layout()
# Save the plot as a PNG file
fig.savefig("delta_proportion_boxplot.png")
plt.show()


# List of LSOAs
lsoas = list(counts_diff_all.keys())

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
global_min = min(min(counts_diff_all[lsoa].values()) for lsoa in lsoas)
global_max = max(max(counts_diff_all[lsoa].values()) for lsoa in lsoas)
global_min_buffer = global_min + 0.1 * global_min
global_max_buffer = global_max + 0.1 * global_max

print(f"global_min_buffer: {global_min_buffer}")
print(f"global_max_buffer: {global_max_buffer}")
# Get the color palette used by seaborn's boxplot
palette = sns.color_palette()
# Loop over the LSOAs
for i, lsoa in enumerate(lsoas):
    # Create a DataFrame for the current LSOA
    df_counts = pd.DataFrame(
        {
            "Tenure": list(counts_diff_all[lsoa].keys()),
            "Difference": list(
                map(float, counts_diff_all[lsoa].values())
            ),  # Convert to float
        }
    )

    # Create a bar chart in the current subplot
    df_counts.plot(
        x="Tenure",
        y="Difference",
        kind="bar",
        ax=axs[i],
        color=[palette[order.index(tenure)] for tenure in df_counts["Tenure"]],
        legend=False,
    )

    # Set the title and labels
    axs[i].set_title(f"$\\Delta_{{count}}$ for {lsoa}")
    axs[i].set_ylabel("$\Delta_{count}$")
    axs[i].set_xlabel("Tenure")
    axs[i].set_xticklabels(df_counts["Tenure"], rotation=15)

    # Set the y-axis limits
    axs[i].set_ylim(global_min_buffer, global_max_buffer)
    # Add a gridline at 0
    axs[i].axhline(0, color="black", linewidth=0.5)

# Remove unused subplots
for i in range(n_lsoas, n_rows * n_cols):
    fig.delaxes(axs[i])

# Show the plot
fig.suptitle(
    r"Pre-weighted $\Delta_{count}(n_{\mathrm{EPC\ properties}} - n_{\mathrm{census\ properties}})$",
    y=0.98,
)
plt.tight_layout()
plt.show()
fig.savefig("preweighted_delta_count_LSOAs.png")

# List of LSOAs
lsoas = list(proportions_diff_all.keys())

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
global_min = min(min(proportions_diff_all[lsoa].values()) for lsoa in lsoas)
global_max = max(max(proportions_diff_all[lsoa].values()) for lsoa in lsoas)
global_min_buffer = global_min + 0.1 * global_min
global_max_buffer = global_max + 0.1 * global_max

# Get the color palette used by seaborn's boxplot
palette = sns.color_palette()

# Loop over the LSOAs
for i, lsoa in enumerate(lsoas):
    # Create a DataFrame for the current LSOA
    df_proportions = pd.DataFrame(
        {
            "Tenure": list(proportions_diff_all[lsoa].keys()),
            "Difference": list(
                map(float, proportions_diff_all[lsoa].values())
            ),  # Convert to float
        }
    )

    # Create a bar chart in the current subplot
    df_proportions.plot(
        x="Tenure",
        y="Difference",
        kind="bar",
        ax=axs[i],
        color=[palette[order.index(tenure)] for tenure in df_proportions["Tenure"]],
        legend=False,
    )

    # Set the title and labels
    axs[i].set_title(f"$\\Delta_{{proportion}}$ for {lsoa}")
    axs[i].set_ylabel("$\Delta_{proportion}$")
    axs[i].set_xlabel("Tenure")
    axs[i].set_xticklabels(df_proportions["Tenure"], rotation=15)

    # Set the y-axis limits
    axs[i].set_ylim(global_min_buffer, global_max_buffer)

    # Add a gridline at 0
    axs[i].axhline(0, color="black", linewidth=0.5)

# Remove unused subplots
for i in range(n_lsoas, n_rows * n_cols):
    fig.delaxes(axs[i])

# Show the plot
fig.suptitle(r"Pre-weighted $\Delta_{proportion}$", y=0.98)

plt.tight_layout()
plt.show()
fig.savefig("preweighted_delta_proportion_LSOAs.png")


# Now we want to save an OUTPUT file with the results of this basic analysis.
