# %%
from asf_heat_pump_suitability.getters.s3_getters import load_s3_data
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target

# %%
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load results - update paths as required
s_results = load_s3_data(
    "asf-heat-pump-suitability",
    "evaluation/reweighting/2023Q4/20250103_2023_Q4_EPC_weights_evaluation_S.json",
)
ew_results = load_s3_data(
    "asf-heat-pump-suitability",
    "evaluation/reweighting/2023Q4/20250103_2023_Q4_EPC_weights_evaluation_EW.json",
)

# %% [markdown]
# ## Restructure data


# %%
def restructure_df_data(results: dict, scotland: bool) -> pd.DataFrame:
    """
    Restructure results data from dict into dataframe.

    Args:
        results (dict): reweighting evaluation results
        scotland (bool): True if evaluating Scotland results, False for England and Wales

    Returns:
        pd.DataFrame: reweighting evaluation results in dataframe
    """

    metric_name = "mae_no_missing_cats"

    if scotland:
        feature_names = ["tenure", "property_type"]
    else:
        feature_names = ["tenure", "property_type", "build_year"]

    mae_unweight_per_feature = defaultdict(list)
    mae_reweight_per_feature = defaultdict(list)
    mae_diff_per_feature = defaultdict(list)
    num_props_per_feature = defaultdict(list)
    error_red_per_feature = defaultdict(list)
    for feature_name in feature_names:
        for lsoa, lsoa_results in results.items():
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
                num_props_per_feature[feature_name].append(lsoa_results["n_properties"])
            else:
                mae_unweight_per_feature[feature_name].append(None)
                mae_reweight_per_feature[feature_name].append(None)
                error_red_per_feature[feature_name].append(None)
                mae_diff_per_feature[feature_name].append(None)

    feature_metrics = {
        "lsoa": list(results.keys()),
        "num_properties": num_props_per_feature["tenure"],
    }

    for feature in feature_names:
        feature_metrics.update(
            {
                f"{feature}_mae_unweight": mae_unweight_per_feature[feature],
                f"{feature}_mae_reweight": mae_reweight_per_feature[feature],
                f"{feature}_error_red": error_red_per_feature[feature],
                f"{feature}_mae_all": mae_diff_per_feature[feature],
            }
        )

    results_df = pd.DataFrame(feature_metrics)
    results_df["mae_all_average"] = results_df[
        [f"{feature}_mae_all" for feature in feature_names]
    ].mean(axis=1)

    return results_df


# %%
s_results_df = restructure_df_data(s_results, scotland=True)
ew_results_df = restructure_df_data(ew_results, scotland=False)


# %%
def print_results(results_df: pd.DataFrame, scotland: bool):
    """
    Args:
        results_df (pd.DataFrame): reweighting evaluation results
        scotland (bool): True if evaluating Scotland results, False for England and Wales
    """
    if scotland:
        feature_names = ["tenure", "property_type"]
    else:
        feature_names = ["tenure", "property_type", "build_year"]

    for feature in feature_names:
        print(
            f"The reweighting produces better {feature} proportions MAE {round(sum(results_df[f'{feature}_mae_all']>0)*100/len(results_df),2)}% of the time"
        )
        print(
            f"The reweighting produces the same {feature} proportions MAE {round(sum(results_df[f'{feature}_mae_all']==0)*100/len(results_df),2)}% of the time"
        )


# %% [markdown]
# ## Scotland

# %%
print_results(s_results_df, scotland=True)

# %%
# Average error reduction after reweighting per feature, per reweighting model
features = ["property_type", "tenure"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
fig.tight_layout(h_pad=5)
fontsize = 10

for j, feature_name in enumerate(features):
    axs[j].hist(
        [m for m in s_results_df[f"{feature_name}_mae_all"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(
        f"{feature_name}\nMAE reduction after reweighting", fontsize=fontsize
    )

# %%
# Average error after reweighting per feature, per reweighting model
features = ["property_type", "tenure"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
fig.tight_layout(h_pad=5)
fontsize = 10

for j, feature_name in enumerate(features):
    axs[j].hist(
        [m for m in s_results_df[f"{feature_name}_mae_reweight"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(f"{feature_name}\nMAE after reweighting", fontsize=fontsize)

# %%
# Average error before reweighting per feature
features = ["property_type", "tenure"]
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
fig.tight_layout(h_pad=5)
fontsize = 10


for j, feature_name in enumerate(features):

    axs[j].hist(
        [m for m in s_results_df[f"{feature_name}_mae_unweight"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(f"{feature_name}\nMAE before weighting", fontsize=fontsize)

# %%
# MAE reweighted vs unweighted
fig, axs = plt.subplots(1, 2, figsize=(11, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.3
cmap = "hot"
fontsize = 10

all_results_df_sample = s_results_df.sample(
    5000, random_state=42
)  # it's too hard to plot the original size


def make_scatter(feature_name, axs, ax_i):

    x = [m for m in all_results_df_sample[feature_name + "_mae_unweight"].tolist() if m]
    y = [
        m
        for i, m in enumerate(
            all_results_df_sample[feature_name + "_mae_reweight"].tolist()
        )
        if all_results_df_sample[feature_name + "_mae_unweight"].tolist()[i]
    ]
    c = [
        m
        for i, m in enumerate(all_results_df_sample["num_properties"].tolist())
        if all_results_df_sample[feature_name + "_mae_unweight"].tolist()[i]
    ]

    axs[ax_i].scatter(
        x=x,
        y=y,
        alpha=alpha_val,
        c=c,
        cmap=cmap,
    )
    axs[ax_i].set_xlabel("MAE unweighted")
    axs[ax_i].set_ylabel("MAE reweighted")
    axs[ax_i].set_title(
        f"MAEs for {feature_name} for {len(all_results_df_sample)} LSOAs",
        fontsize=fontsize,
    )
    axs[ax_i].plot(
        [
            all_results_df_sample[feature_name + "_mae_unweight"].min(),
            all_results_df_sample[feature_name + "_mae_unweight"].max(),
        ],
        [
            all_results_df_sample[feature_name + "_mae_unweight"].min(),
            all_results_df_sample[feature_name + "_mae_unweight"].max(),
        ],
        color="#F6A4B7",
    )
    return axs


axs = make_scatter("tenure", axs, 0)
axs = make_scatter("property_type", axs, 1)

# %%
# Scotland has fewer properties per DZ on average than England and Wales have per LSOA
plt.hist(
    s_results_df["num_properties"],
    bins=50,
    color="#0000FF",
)
plt.title("Number of properties reweighted per DataZone in Scotland")
plt.xlabel("Number of properties")
plt.ylabel("Number of DataZones")
plt.show()

# %% [markdown]
# ## England and Wales

# %%
plt.hist(
    ew_results_df["num_properties"],
    bins=50,
    color="#0000FF",
)
plt.title("Number of properties reweighted per LSOA in England and Wales")
plt.xlabel("Number of properties")
plt.ylabel("Number of DataZones")
plt.show()

# %%
print_results(ew_results_df, scotland=False)

# %%
# Average error reduction after reweighting per feature, per reweighting model
features = ["property_type", "tenure", "build_year"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.tight_layout(h_pad=5)
fontsize = 10

for j, feature_name in enumerate(features):
    axs[j].hist(
        [m for m in ew_results_df[f"{feature_name}_mae_all"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(
        f"{feature_name}\nMAE reduction after reweighting", fontsize=fontsize
    )

# %%
# Average error after reweighting per feature, per reweighting model
features = ["property_type", "tenure", "build_year"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.tight_layout(h_pad=5)
fontsize = 10

for j, feature_name in enumerate(features):
    axs[j].hist(
        [m for m in ew_results_df[f"{feature_name}_mae_reweight"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(f"{feature_name}\nMAE after reweighting", fontsize=fontsize)

# %%
# Average error before reweighting per feature
features = ["property_type", "tenure", "build_year"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
fig.tight_layout(h_pad=5)
fontsize = 10


for j, feature_name in enumerate(features):

    axs[j].hist(
        [m for m in ew_results_df[f"{feature_name}_mae_unweight"].tolist() if m],
        bins=50,
        color="#0000FF",
    )
    axs[j].axvline(x=0, color="#F6A4B7", label="axvline - full height")
    axs[j].set_title(f"{feature_name}\nMAE before weighting", fontsize=fontsize)

# %% [markdown]
# ### Additional visuals for England and Wales

# %%
corr = ew_results_df[
    [
        "tenure_mae_all",
        "tenure_error_red",
        "property_type_mae_all",
        "property_type_error_red",
        "build_year_mae_all",
        "build_year_error_red",
        "num_properties",
        "mae_all_average",
    ]
].corr()
corr.style.background_gradient(cmap="coolwarm")

# %%
# MAE_diff_and_error_reduction
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = ew_results_df.sample(
    5000, random_state=42
)  # it's too hard to plot the original size


def make_scatter(feature_name, axs, ax_i):
    x = [m for m in all_results_df_sample[feature_name + "_mae_all"].tolist() if m]
    y = [
        m
        for i, m in enumerate(
            all_results_df_sample[feature_name + "_error_red"].tolist()
        )
        if all_results_df_sample[feature_name + "_mae_all"].tolist()[i]
    ]

    axs[ax_i].scatter(x=x, y=y, alpha=alpha_val, color="#0000FF")
    axs[ax_i].set_yscale("log")
    axs[ax_i].set_xlabel("MAE difference")
    axs[ax_i].set_ylabel("Mean proportion error reduction")
    axs[ax_i].set_title(
        f"{feature_name} for {len(all_results_df_sample)} LSOAs", fontsize=fontsize
    )
    axs[ax_i].axvline(x=0, color="#F6A4B7")
    axs[ax_i].axhline(y=1, color="#F6A4B7")
    return axs


axs = make_scatter("tenure", axs, 0)
axs = make_scatter("property_type", axs, 1)
axs = make_scatter("build_year", axs, 2)

# %%
# MAE_diff_and_num_properties
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = ew_results_df.sample(
    5000, random_state=42
)  # it's too hard to plot the original size


def make_scatter(feature_name, axs, ax_i):
    x = [m for m in all_results_df_sample[feature_name + "_mae_all"].tolist() if m]
    y = [
        m
        for i, m in enumerate(all_results_df_sample["num_properties"].tolist())
        if all_results_df_sample[feature_name + "_mae_all"].tolist()[i]
    ]

    axs[ax_i].scatter(x=x, y=y, alpha=alpha_val, color="#0000FF")
    axs[ax_i].set_xlabel("MAE difference")
    axs[ax_i].set_ylabel("Number of properties")
    axs[ax_i].set_title(
        f"{feature_name} for {len(all_results_df_sample)} LSOAs", fontsize=fontsize
    )
    axs[ax_i].axvline(x=0, color="#F6A4B7")
    return axs


axs = make_scatter("tenure", axs, 0)
axs = make_scatter("property_type", axs, 1)
axs = make_scatter("build_year", axs, 2)

# %%
# Error_reduction_and_num_properties
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = ew_results_df.sample(
    5000, random_state=42
)  # it's too hard to plot the original size


def make_scatter(feature_name, axs, ax_i):
    x = [m for m in all_results_df_sample[feature_name + "_error_red"].tolist() if m]
    y = [
        m
        for i, m in enumerate(all_results_df_sample["num_properties"].tolist())
        if all_results_df_sample[feature_name + "_error_red"].tolist()[i]
    ]

    axs[ax_i].scatter(x=x, y=y, alpha=alpha_val, color="#0000FF")
    axs[ax_i].set_xscale("log")
    axs[ax_i].set_xlabel("Mean proportion error reduction")
    axs[ax_i].set_ylabel("Number of properties")
    axs[ax_i].set_title(
        f"{feature_name} for {len(all_results_df_sample)} LSOAs", fontsize=fontsize
    )
    axs[ax_i].axvline(x=1, color="#F6A4B7")
    return axs


axs = make_scatter("tenure", axs, 0)
axs = make_scatter("property_type", axs, 1)
axs = make_scatter("build_year", axs, 2)

# %%
# MAE reweighted vs unweighted
fig, axs = plt.subplots(1, 3, figsize=(11, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.3
cmap = "hot"
fontsize = 10

all_results_df_sample = ew_results_df.sample(
    5000, random_state=42
)  # it's too hard to plot the original size


def make_scatter(feature_name, axs, ax_i):

    x = [m for m in all_results_df_sample[feature_name + "_mae_unweight"].tolist() if m]
    y = [
        m
        for i, m in enumerate(
            all_results_df_sample[feature_name + "_mae_reweight"].tolist()
        )
        if all_results_df_sample[feature_name + "_mae_unweight"].tolist()[i]
    ]
    c = [
        m
        for i, m in enumerate(all_results_df_sample["num_properties"].tolist())
        if all_results_df_sample[feature_name + "_mae_unweight"].tolist()[i]
    ]

    axs[ax_i].scatter(
        x=x,
        y=y,
        alpha=alpha_val,
        c=c,
        cmap=cmap,
    )
    axs[ax_i].set_xlabel("MAE unweighted")
    axs[ax_i].set_ylabel("MAE reweighted")
    axs[ax_i].set_title(
        f"MAEs for {feature_name} for {len(all_results_df_sample)} LSOAs",
        fontsize=fontsize,
    )
    axs[ax_i].plot(
        [
            all_results_df_sample[feature_name + "_mae_unweight"].min(),
            all_results_df_sample[feature_name + "_mae_unweight"].max(),
        ],
        [
            all_results_df_sample[feature_name + "_mae_unweight"].min(),
            all_results_df_sample[feature_name + "_mae_unweight"].max(),
        ],
        color="#F6A4B7",
    )
    return axs


axs = make_scatter("tenure", axs, 0)
axs = make_scatter("property_type", axs, 1)
axs = make_scatter("build_year", axs, 2)

# %%
