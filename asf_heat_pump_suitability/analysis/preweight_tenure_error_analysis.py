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
