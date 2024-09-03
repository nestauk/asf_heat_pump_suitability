"""
This script performs an error analysis on the tenure data by LSOA (Lower Layer Super Output Area).

First, it loads the sample and target data. Then, it processes all LSOAs to get error metrics and differences in counts and proportions. These metrics are saved to a CSV file.

Next, it creates and melts dataframes for counts and proportions differences, extracts values from these melted dataframes, and converts them to pandas dataframes.

Finally, it creates boxplots and difference plots for counts and proportions differences. These plots provide a visual representation of the differences in counts and proportions by tenure for each LSOA.

This script is intended to be run as a standalone script.
"""

import s3fs
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.evaluation.evaluate_reweighting import *


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
        target (pl.DataFrame): The preprocessed census data, with irrelevant rows and columns removed, and columns renamed.
    """
    fs = s3fs.S3FileSystem()
    with fs.open(target_path, mode="rb") as f:
        content = f.read()
    target = pl.read_excel(content, sheet_name="3c", engine="calamine")
    target = target.rename(target[2].to_dicts().pop()).slice(3)
    target = target.rename(
        {
            "Area Code": "lsoa",
            "Owned or shared ownership": "owner-occupied",
            "Social Rented": "rental (social)",
            "Private Rented or lives rent free": "rental (private)",
        }
    )
    censored_vals = 0
    int_cols = ["owner-occupied", "rental (private)", "rental (social)"]
    target = target.with_columns(
        [pl.col(int_cols).str.replace("c", f"{censored_vals}").cast(pl.Int64)]
    ).select(pl.col(["lsoa", "owner-occupied", "rental (private)", "rental (social)"]))
    return target


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
            # TODO: unresolved reference to total_counts - is this meant to be a param?
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


if __name__ == "__main__":
    sample_path = "s3://asf-heat-pump-suitability/outputs/epc_sample_lsoa.parquet"
    target_path = config["data_source"]["EW_housing_characteristics_census"]
    DELTA_COUNT = "Delta count"
    DELTA_PROPORTION = "Delta proportion"
    ORDER = ["owner-occupied", "rental (private)", "rental (social)"]
    LSOA = "lsoa"
    TARGET_FEATURE = "tenure"
    test_lsoas = ["W01000328", "E01033942", "E01012376", "W01000527", "E01002058"]
    number_of_lsoas = len(test_lsoas)
    # Load the sample and target data
    sample = load_sample(sample_path)
    tenure = load_target(target_path)

    # Process all LSOAs and get error metrics and differences
    (
        rmse_all_without_missing_categories,
        mae_all_without_missing_categories,
        rmse_all_with_missing_categories,
        mae_all_with_missing_categories,
        counts_diff_all,
        proportions_diff_all,
        total_counts,
    ) = process_all_lsoas(sample, tenure, test_lsoas, TARGET_FEATURE)

    # Save the error metrics to a CSV file
    create_error_dataframe_and_save(
        rmse_all_without_missing_categories,
        mae_all_without_missing_categories,
        rmse_all_with_missing_categories,
        mae_all_with_missing_categories,
        "tenure_lsoa_errors_sklearn.csv",
    )

    # Create and melt dataframes for counts and proportions differences
    counts_df_melted = create_dataframe_and_melt(
        counts_diff_all, DELTA_COUNT, TARGET_FEATURE, LSOA
    )
    proportions_df_melted = create_dataframe_and_melt(
        proportions_diff_all, DELTA_PROPORTION, TARGET_FEATURE, LSOA
    )

    # Extract values from the melted dataframes
    melted_rows = extract_values(counts_df_melted, DELTA_COUNT, TARGET_FEATURE, LSOA)
    counts_df_extracted = pl.DataFrame(melted_rows)
    proportion_melted_rows = extract_values(
        proportions_df_melted, DELTA_PROPORTION, TARGET_FEATURE, LSOA
    )
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
        target_feature_name=TARGET_FEATURE,
        order_preference=ORDER,
    )
    create_boxplot(
        proportions_df_extracted_pd,
        DELTA_PROPORTION,
        f"Boxplot of pre-weighted $\\Delta_{{proportion}}$ by Tenure for $n_{{LSOA}}={number_of_lsoas}$",
        "delta_proportion_boxplot.png",
        target_feature_name=TARGET_FEATURE,
        order_preference=ORDER,
    )

    # Create difference plots for counts and proportions differences
    create_difference_plots(
        counts_diff_all,
        r"Pre-weighted $\Delta_{count}(n_{\mathrm{EPC\ properties}} - n_{\mathrm{census\ properties}})$",
        "preweighted_delta_count_LSOAs.png",
        "count",
        target_feature_name=TARGET_FEATURE,
        order_preference=ORDER,
    )
    create_difference_plots(
        proportions_diff_all,
        r"Pre-weighted $\Delta_{proportion}$",
        "preweighted_delta_proportion_LSOAs.png",
        "proportion",
        target_feature_name=TARGET_FEATURE,
        order_preference=ORDER,
    )
