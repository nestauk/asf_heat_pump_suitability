# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: asf_heat_pump_suitability
#     language: python
#     name: asf_heat_pump_suitability
# ---

# %%
from asf_heat_pump_suitability.getters.s3_getters import load_s3_data
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target
from asf_heat_pump_suitability import PROJECT_DIR


# %%
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# Load results
results_3f = load_s3_data(
    "asf-heat-pump-suitability",
    "outputs/2023Q4/2023_Q4_EPC_weights_3_features_evaluation.json",
)
results_2f = load_s3_data(
    "asf-heat-pump-suitability",
    "outputs/2023Q4/2023_Q4_EPC_weights_2_features_evaluation.json",
)
results_3f_mixed = load_s3_data(
    "asf-heat-pump-suitability",
    "outputs/2023Q4/2023_Q4_EPC_weights_3_features_mixed_lsoa_la_evaluation.json",
)


# %%
output_dir = f"{PROJECT_DIR}/outputs/figures/reweighting_evaluation"


# %% [markdown]
# ## Restructure data


# %%
def restructure_df_data(full_results: dict) -> pd.DataFrame:
    """
    Restructure results data from dict into dataframe.

    Args:
        full_results (dict): reweighting evaluation results

    Returns:
        pd.DataFrame: reweighting evaluation results in dataframe
    """

    metric_name = "mae_no_missing_cats"

    evaluation_feature_names = ["tenure", "property_type", "build_year"]

    mae_unweight_per_feature = defaultdict(list)
    mae_reweight_per_feature = defaultdict(list)
    mae_diff_per_feature = defaultdict(list)
    num_props_per_feature = defaultdict(list)
    error_red_per_feature = defaultdict(list)
    for feature_name in evaluation_feature_names:
        for lsoa, lsoa_results in full_results.items():
            if lsoa_results[feature_name]:
                r = lsoa_results[feature_name][metric_name]
                mae_unweight_per_feature[feature_name].append(r["unweight"])
                mae_reweight_per_feature[feature_name].append(r["reweight"])
                error_red_per_feature[feature_name].append(r["error_reduction"])
                if r["unweight"] and r["reweight"]:
                    diff = r["unweight"] - r["reweight"]
                    mae_diff_per_feature[feature_name].append(diff)
                else:
                    mae_diff_per_feature[feature_name].append(None)
                num_props_per_feature[feature_name].append(lsoa_results["num_props"])
            else:
                mae_unweight_per_feature[feature_name].append(None)
                mae_reweight_per_feature[feature_name].append(None)
                error_red_per_feature[feature_name].append(None)
                mae_diff_per_feature[feature_name].append(None)

    all_results_df = pd.DataFrame(
        {
            "lsoa": list(full_results.keys()),
            "tenure_mae_unweight": mae_unweight_per_feature["tenure"],
            "tenure_mae_reweight": mae_reweight_per_feature["tenure"],
            "tenure_error_red": error_red_per_feature["tenure"],
            "tenure_mae_all": mae_diff_per_feature["tenure"],
            "property_type_mae_unweight": mae_unweight_per_feature["property_type"],
            "property_type_mae_reweight": mae_reweight_per_feature["property_type"],
            "property_type_error_red": error_red_per_feature["property_type"],
            "property_type_mae_all": mae_diff_per_feature["property_type"],
            "build_year_mae_unweight": mae_unweight_per_feature["build_year"],
            "build_year_mae_reweight": mae_reweight_per_feature["build_year"],
            "build_year_error_red": error_red_per_feature["build_year"],
            "build_year_mae_all": mae_diff_per_feature["build_year"],
            "num_properties": num_props_per_feature["tenure"],
        }
    )
    all_results_df["mae_all_average"] = all_results_df[
        ["tenure_mae_all", "property_type_mae_all", "build_year_mae_all"]
    ].mean(axis=1)

    return all_results_df


# %%
results_dfs = {
    "3 features": restructure_df_data(results_3f),
    "2 features": restructure_df_data(results_2f),
    "3 features (mixed-level)": restructure_df_data(results_3f_mixed),
}


# %% [markdown]
# ## Analysis


# %%
def print_results(all_results_df: pd.DataFrame):
    print(
        f"The reweighting produces better Tenure proportions MAE {round(sum(all_results_df['tenure_mae_all']>0)*100/len(all_results_df),2)}% of the time"
    )
    print(
        f"The reweighting produces the same Tenure proportions MAE {round(sum(all_results_df['tenure_mae_all']==0)*100/len(all_results_df),2)}% of the time"
    )

    print(
        f"The reweighting produces better Property Type proportions MAE {round(sum(all_results_df['property_type_mae_all']>0)*100/len(all_results_df),2)}% of the time"
    )
    print(
        f"The reweighting produces the same Property Type proportions MAE {round(sum(all_results_df['property_type_mae_all']==0)*100/len(all_results_df),2)}% of the time"
    )

    print(
        f"The reweighting produces better Build Year proportions MAE {round(sum(all_results_df['build_year_mae_all']>0)*100/len(all_results_df),2)}% of the time"
    )
    print(
        f"The reweighting produces the same Build Year proportions MAE {round(sum(all_results_df['build_year_mae_all']==0)*100/len(all_results_df),2)}% of the time"
    )


# %%
print_results(results_dfs["3 features"])

# %%
print_results(results_dfs["2 features"])

# %%
print_results(results_dfs["3 features (mixed-level)"])

# %%
features = ["property_type", "tenure", "build_year"]
fig, axs = plt.subplots(len(results_dfs), 3, figsize=(12, 3 * len(results_dfs)))
fig.tight_layout(h_pad=5)
fontsize = 10

for i, (composition, results_df) in enumerate(results_dfs.items()):
    for j, feature_name in enumerate(features):

        # feature_name = "tenure"
        axs[i][j].hist(
            [m for m in results_df[f"{feature_name}_mae_all"].tolist() if m],
            bins=50,
            color="#0000FF",
        )
        axs[i][j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
        axs[i][j].set_title(
            f"{feature_name}\nMAE diff for reweighting with {composition}",
            fontsize=fontsize,
        )

    fig.savefig(
        f"{output_dir}/MAE_diff_per_feature_histograms.png", bbox_inches="tight"
    )

# %%
