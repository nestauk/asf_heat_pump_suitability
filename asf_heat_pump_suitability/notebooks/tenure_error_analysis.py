import random
import polars as pl
import balance
import s3fs
from asf_heat_pump_suitability import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

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
# Initialize dictionaries to store the results
counts_diff_all = {}
proportions_diff_all = {}

# Initialize the dictionary
total_counts = {}
# Iterate over all LSOAs
for test_lsoa in test_lsoas:
    # Get the subset of the sample and the tenure data for the current LSOA
    subset = sample.filter(pl.col("lsoa") == test_lsoa)
    tenure_subset = tenure.filter(pl.col("lsoa") == test_lsoa)

    # Perform the same calculations as before
    counts_subset_epc_sample = subset.group_by("tenure").agg(
        [pl.col("tenure").count().alias("count")]
    )
    counts = dict(
        zip(counts_subset_epc_sample["tenure"], counts_subset_epc_sample["count"])
    )
    total = sum(counts.values())
    proportions = {k: v / total for k, v in counts.items()}

    tenure_dict = tenure_subset.to_dicts()[0]
    del tenure_dict["lsoa"]
    total_lsoa = sum(tenure_dict.values())
    print("total lsoa value:")
    print(total_lsoa)
    tenure_dict_proportions = {k: v / total_lsoa for k, v in tenure_dict.items()}

    counts_diff = {
        k: counts.get(k, 0) - tenure_dict.get(k, 0)
        for k in set(counts) | set(tenure_dict)
    }
    proportions_diff = {
        k: proportions.get(k, 0) - tenure_dict_proportions.get(k, 0)
        for k in set(proportions) | set(tenure_dict_proportions)
    }

    # Store the results in the dictionaries
    counts_diff_all[test_lsoa] = counts_diff
    proportions_diff_all[test_lsoa] = proportions_diff
    # Update the total counts dictionary
    for k, v in counts.items():
        if k in total_counts:
            total_counts[k] += v
        else:
            total_counts[k] = v
# Print the total counts
print("total counts")
print(total_counts)
# Now counts_diff_all and proportions_diff_all contain the differences in counts and proportions for all LSOAs
print(counts_diff_all)
print(proportions_diff_all)

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

print(f"global_min_buffer: {global_min_buffer}")
print(f"global_max_buffer: {global_max_buffer}")

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
