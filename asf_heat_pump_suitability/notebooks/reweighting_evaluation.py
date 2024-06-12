#!/usr/bin/env python
# coding: utf-8

# In[10]:


from asf_heat_pump_suitability.getters.s3_getters import load_s3_data
from asf_heat_pump_suitability.pipeline.reweight_epc import prepare_target
from asf_heat_pump_suitability import PROJECT_DIR


# In[2]:


from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


full_results = load_s3_data(
    "asf-heat-pump-suitability", "outputs/2023_Q2_EPC_enhanced_weights_evaluation.json"
)


# In[4]:


len(full_results)


# In[15]:


output_dir = f"{PROJECT_DIR}/outputs/figures/reweighting_evaluation"


# ## Restructure data

# In[6]:


metric_name = "mae_all_cats"

evaluation_feature_names = ["tenure", "property_type", "build_year", "nrooms"]

mae_unweight_per_feature = defaultdict(list)
mae_reweight_per_feature = defaultdict(list)
mae_diff_per_feature = defaultdict(list)
num_props_per_feature = defaultdict(list)
error_red_per_feature = defaultdict(list)
for feature_name in evaluation_feature_names:
    for lsoa, lsoa_results in full_results.items():
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


# In[7]:


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
        "nrooms_mae_unweight": mae_unweight_per_feature["nrooms"],
        "nrooms_mae_reweight": mae_reweight_per_feature["nrooms"],
        "nrooms_error_red": error_red_per_feature["nrooms"],
        "nrooms_mae_all": mae_diff_per_feature["nrooms"],
        "num_properties": num_props_per_feature["tenure"],
    }
)
all_results_df["mae_all_average"] = all_results_df[
    ["tenure_mae_all", "property_type_mae_all", "build_year_mae_all"]
].mean(axis=1)


# ## Analysis

# In[8]:


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

print(
    f"The reweighting produces better nrooms proportions MAE {round(sum(all_results_df['nrooms_mae_all']>0)*100/len(all_results_df),2)}% of the time"
)
print(
    f"The reweighting produces the same nrooms proportions MAE {round(sum(all_results_df['nrooms_mae_all']==0)*100/len(all_results_df),2)}% of the time"
)


# In[57]:


corr = all_results_df[
    [
        "tenure_mae_all",
        "tenure_error_red",
        "property_type_mae_all",
        "property_type_error_red",
        "build_year_mae_all",
        "build_year_error_red",
        "nrooms_mae_all",
        "num_properties",
        "mae_all_average",
    ]
].corr()
corr.to_csv(f"{output_dir}/MAE_diff_correlation_matrix.csv")
corr.style.background_gradient(cmap="coolwarm")


# In[28]:


fig, axs = plt.subplots(1, 4, figsize=(12, 3))
fig.tight_layout(h_pad=10)
fontsize = 10

feature_name = "tenure"
axs[0].hist(
    [m for m in all_results_df[f"{feature_name}_mae_all"].tolist() if m],
    bins=50,
    color="#0000FF",
)
axs[0].axvline(x=0, color="#F6A4B7", label="axvline - full height")
axs[0].set_title(
    f"{feature_name}\nMAE diff for {len(full_results)} LSOAs", fontsize=fontsize
)

feature_name = "property_type"
axs[1].hist(
    [m for m in all_results_df[f"{feature_name}_mae_all"].tolist() if m],
    bins=50,
    color="#0000FF",
)
axs[1].axvline(x=0, color="#F6A4B7", label="axvline - full height")
axs[1].set_title(
    f"{feature_name}\nMAE diff for {len(full_results)} LSOAs", fontsize=fontsize
)

feature_name = "build_year"
axs[2].hist(
    [m for m in all_results_df[f"{feature_name}_mae_all"].tolist() if m],
    bins=50,
    color="#0000FF",
)
axs[2].axvline(x=0, color="#F6A4B7", label="axvline - full height")
axs[2].set_title(
    f"{feature_name}\nMAE diff for {len(full_results)} LSOAs", fontsize=fontsize
)

feature_name = "nrooms"
axs[3].hist(
    [m for m in all_results_df[f"{feature_name}_mae_all"].tolist() if m],
    bins=50,
    color="#0000FF",
)
axs[3].axvline(x=0, color="#F6A4B7", label="axvline - full height")
axs[3].set_title(
    f"{feature_name}\nMAE diff for {len(full_results)} LSOAs", fontsize=fontsize
)

fig.savefig(f"{output_dir}/MAE_diff_per_feature_histograms.png", bbox_inches="tight")


# In[41]:


fig, axs = plt.subplots(1, 4, figsize=(14, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = all_results_df.sample(
    10000, random_state=42
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
axs = make_scatter("nrooms", axs, 3)

fig.savefig(f"{output_dir}/MAE_diff_and_error_reduction.png", bbox_inches="tight")


# In[46]:


fig, axs = plt.subplots(1, 4, figsize=(14, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = all_results_df.sample(
    10000, random_state=42
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
axs = make_scatter("nrooms", axs, 3)

fig.savefig(f"{output_dir}/MAE_diff_and_num_properties.png", bbox_inches="tight")


# In[55]:


fig, axs = plt.subplots(1, 4, figsize=(14, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.01
fontsize = 10

all_results_df_sample = all_results_df.sample(
    10000, random_state=42
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
axs = make_scatter("nrooms", axs, 3)

fig.savefig(f"{output_dir}/error_reduction_and_num_properties.png", bbox_inches="tight")


# In[51]:


fig, axs = plt.subplots(1, 4, figsize=(14, 3))
fig.tight_layout(h_pad=4, w_pad=6)

alpha_val = 0.3
cmap = "hot"
fontsize = 10

all_results_df_sample = all_results_df.sample(
    10000, random_state=42
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
axs = make_scatter("nrooms", axs, 3)

fig.savefig(f"{output_dir}/MAE_unweigh_reweight.png", bbox_inches="tight")


# In[ ]:
