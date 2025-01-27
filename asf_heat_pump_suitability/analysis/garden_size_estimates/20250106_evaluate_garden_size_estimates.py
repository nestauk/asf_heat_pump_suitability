# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from asf_heat_pump_suitability.pipeline.prepare_features import (
    garden_space_avg,
    household_count,
)

# %% [markdown]
# ## Compare calculated garden size to avg
#
# ### 1. Load average garden per MSOA

# %%
avg_gardens_df = garden_space_avg.generate_df_garden_space_avg()

# %% [markdown]
# ### 2. Load garden size estimates

# %%
gardens_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/20250115_2023_Q4_EPC_garden_size_estimates_EWS_deduplicated_FOR_EVALUATION_ONLY.parquet"
)

# %%
# Get counts of UPRN per garden
uprn_count = gardens_df.group_by("NATIONALCADASTRALREFERENCE").agg(
    pl.col("UPRN").count().alias("UPRN_count")
)

# Assign 1 garden size per cadastral
cadastral_garden_size = gardens_df.group_by("NATIONALCADASTRALREFERENCE").agg(
    pl.col("garden_area_m2").first().alias("cadastral_garden_size_m2")
)

# Join to garden size df
gardens_df = gardens_df.join(
    uprn_count, how="left", on="NATIONALCADASTRALREFERENCE"
).join(cadastral_garden_size, how="left", on="NATIONALCADASTRALREFERENCE")

# Divide shared gardens equally among UPRNs sharing gardens
gardens_df = gardens_df.with_columns(
    (pl.col("cadastral_garden_size_m2") / pl.col("UPRN_count")).alias(
        "divided_garden_area_m2"
    )
)

# %% [markdown]
# ### 3. Join EPC with gardens and MSOA averages

# %%
# Features data
raw_epc_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/20250120_2023_Q4_EPC_features.parquet",
    columns=[
        "UPRN",
        "msoa",
        "property_type",
        "msoa_avg_outdoor_space_property_type",
        "COUNTRY",
    ],
)

# %%
# Join garden estimates and garden averages to EPC
epc_df = raw_epc_df.join(gardens_df, how="left", on="UPRN").join(
    avg_gardens_df,
    how="left",
    left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
    right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
)

# %%
# Filter to rows with a garden and remove top 1% of estimates
epc_df = epc_df.filter(
    pl.col("msoa_avg_outdoor_space_m2").is_not_null(),
    pl.col("divided_garden_area_m2").is_not_null(),
    pl.col("divided_garden_area_m2") > 0,
    pl.col("divided_garden_area_m2") < pl.col("divided_garden_area_m2").quantile(0.99),
)

# %%
# Calculate mean estimates per MSOA
results = epc_df.group_by(["msoa", "msoa_avg_outdoor_space_property_type"]).agg(
    pl.col("divided_garden_area_m2").mean().alias("mean_divided_garden_area_m2"),
    pl.col("divided_garden_area_m2").count().alias("n_properties"),
    pl.col("msoa_avg_outdoor_space_m2").first(),
)

# Filter results to MSOAs with averages calculated from 15 or more properties
results = results.filter(pl.col("n_properties") >= 15)

# Get errors per MSOA
results = results.with_columns(
    (pl.col("msoa_avg_outdoor_space_m2") - pl.col("mean_divided_garden_area_m2")).alias(
        "diff"
    ),
    (pl.col("msoa_avg_outdoor_space_m2") - pl.col("mean_divided_garden_area_m2"))
    .abs()
    .alias("absolute_error"),
)

# %% [markdown]
# ### 4. Analysis and results

# %%
results["absolute_error"].describe()

# %%
# See summary statistics for each property type
for property_type in results["msoa_avg_outdoor_space_property_type"].unique():
    print(property_type)
    print(
        results.filter(pl.col("msoa_avg_outdoor_space_property_type") == property_type)[
            "diff"
        ].describe()
    )

# %%
# Plot distribution of average MSOA garden size estimate error
bins = np.linspace(results["diff"].min(), results["diff"].max(), 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for i, property_type in enumerate(
    results["msoa_avg_outdoor_space_property_type"].unique()
):
    axes[i].hist(
        results.filter(pl.col("msoa_avg_outdoor_space_property_type") == property_type)[
            "diff"
        ],
        bins=bins,
    )
    axes[i].set_title(property_type)
    axes[i].set_xlabel("Estimated garden size error")
    axes[i].set_ylabel("Count of MSOAs")

fig.suptitle("Garden size errors (ONS MSOA avg minus MSOA avg of estimated sizes)")
plt.show()

# %%
# Plot scatter of average garden size estimate vs ONS values
fig, ax = plt.subplots()
print(len(results))
plot = results.filter(
    pl.col("msoa_avg_outdoor_space_m2") < 2100,
    pl.col("mean_divided_garden_area_m2") < 3500,
)
print(len(plot))
plt.scatter(
    plot["msoa_avg_outdoor_space_m2"],
    plot["mean_divided_garden_area_m2"],
    alpha=0.3,
    s=1,
)
plt.xlabel("ONS MSOA average (m2)")
plt.ylabel("Estimated MSOA average (m2)")
plt.plot(
    np.unique(plot["msoa_avg_outdoor_space_m2"]),
    np.poly1d(
        np.polyfit(
            plot["msoa_avg_outdoor_space_m2"], plot["mean_divided_garden_area_m2"], 1
        )
    )(np.unique(plot["msoa_avg_outdoor_space_m2"])),
    color="red",
    linestyle="--",
    label="Line of best fit",
)
plt.plot(np.arange(0, 2000), np.arange(0, 2000), color="orange", label="Ideal line")
plt.legend()
plt.title("ONS MSOA average garden size vs our estimates (excluding outliers)")

# %%
# Plot scatter of average garden size estimate vs ONS values per property type
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for i, property_type in enumerate(["Houses", "Flats", "unknown"]):
    plot = results.filter(
        pl.col("msoa_avg_outdoor_space_property_type") == property_type
    )
    print(len(plot))
    plot = plot.filter(
        pl.col("msoa_avg_outdoor_space_m2") < 12000,
        pl.col("mean_divided_garden_area_m2") < 4000,
    )
    print(len(plot))
    axes[i].scatter(
        plot["msoa_avg_outdoor_space_m2"],
        plot["mean_divided_garden_area_m2"],
        alpha=0.3,
        s=1,
    )
    axes[i].set_xlabel("ONS MSOA average (m2)")
    axes[i].set_ylabel("Estimated MSOA average (m2)")
    axes[i].set_title(f"{property_type} (MSOA count = {plot['msoa'].n_unique()})")
    axes[i].plot(
        np.unique(plot["msoa_avg_outdoor_space_m2"]),
        np.poly1d(
            np.polyfit(
                plot["msoa_avg_outdoor_space_m2"],
                plot["mean_divided_garden_area_m2"],
                1,
            )
        )(np.unique(plot["msoa_avg_outdoor_space_m2"])),
        color="red",
        linestyle="--",
        label="Line of best fit",
    )
    axes[i].plot(
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        color="orange",
        label="Ideal line",
    )

plt.gca().legend(("MSOA", "Line of best fit", "Ideal line"))
fig.suptitle("ONS MSOA average garden size vs our estimates (excluding outliers)")
plt.show()

# %%
print("RMSE:")
print(
    metrics.root_mean_squared_error(
        results["msoa_avg_outdoor_space_m2"], results["mean_divided_garden_area_m2"]
    )
)
print("Median absolute error:")
print(
    metrics.median_absolute_error(
        results["msoa_avg_outdoor_space_m2"], results["mean_divided_garden_area_m2"]
    )
)

# %% [markdown]
# ### 5. Compare weighted results

# %%
# Filter to overall average garden size. We need to compare all property types to the total average to be able to use the weights
total_avg_gardens_df = avg_gardens_df.filter(
    pl.col("msoa_avg_outdoor_space_property_type") == "unknown"
)

# %%
# Load weights
weights_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/weights/20250102_2023_Q4_EPC_weights.parquet",
    columns=["UPRN", "lsoa", "proportional_weight"],
)

# %%
# Join weights to EPC
epc_df = raw_epc_df.join(weights_df, how="left", on="UPRN")

# %%
# Load household counts per LSOA to multiply weights
n_households = household_count.load_transform_df_n_households()

# %%
# Join MSOA total weights to EPC and calculate proportional weights per MSOA
epc_df = epc_df.join(n_households, how="left", on="lsoa")
epc_df = epc_df.with_columns(
    (pl.col("proportional_weight") * pl.col("households_count")).alias("weight")
)

# %%
# Join garden estimates and average garden size
epc_df = epc_df.join(gardens_df, how="left", on="UPRN").join(
    total_avg_gardens_df, how="left", left_on="msoa", right_on="MSOA code"
)

# %%
# Apply selection criteria
epc_df = epc_df.filter(
    pl.col("msoa_avg_outdoor_space_m2").is_not_null(),
    pl.col("divided_garden_area_m2").is_not_null(),
    pl.col("divided_garden_area_m2") > 0,
    pl.col("divided_garden_area_m2") < pl.col("divided_garden_area_m2").quantile(0.99),
    pl.col("weight").is_not_null(),
)

# %%
# After removing rows, we need to check whether our weights still add approximately to 1 per LSOA
lsoa_weights = (
    epc_df.select(["lsoa", "proportional_weight"])
    .group_by("lsoa")
    .agg(pl.col("proportional_weight").sum().alias("lsoa_weight_total_updated"))
)
lsoa_weights["lsoa_weight_total_updated"].describe()

# %%
# # Filter to rows MSOAs with total weight of at least 0.9 and weight garden estimates
epc_df = epc_df.join(lsoa_weights, how="left", on="lsoa")
epc_df = epc_df.filter(
    pl.col("lsoa_weight_total_updated") >= 0.9, pl.col("lsoa_weight_total_updated") <= 1
)
epc_df = epc_df.with_columns(
    (pl.col("divided_garden_area_m2") * pl.col("weight")).alias(
        "weighted_divided_garden_area_m2"
    )
)

# %%
# Calculate weighted mean garden area per LSOA and filter to LSOAs with at least 15 properties
weighted_results = epc_df.group_by(["lsoa"]).agg(
    pl.col("weighted_divided_garden_area_m2")
    .sum()
    .alias("lsoa_total_weighted_divided_garden_area_m2"),
    pl.col("weight").sum().alias("lsoa_total_weight"),
    pl.col("divided_garden_area_m2").count().alias("n_properties"),
    pl.col("msoa").first(),
    pl.col("msoa_avg_outdoor_space_m2").first(),
)
weighted_results = weighted_results.with_columns(
    (
        pl.col("lsoa_total_weighted_divided_garden_area_m2")
        / pl.col("lsoa_total_weight")
    ).alias("lsoa_weighted_mean_divided_garden_area_m2")
)
weighted_results = weighted_results.filter(pl.col("n_properties") >= 15)

# Calculate MSOA average by averaging LSOA weighted averages
weighted_results = weighted_results.group_by("msoa").agg(
    pl.col("lsoa_weighted_mean_divided_garden_area_m2")
    .mean()
    .alias("msoa_weighted_average_garden_area_m2"),
    pl.col("msoa_avg_outdoor_space_m2").first(),
)
weighted_results["msoa_weighted_average_garden_area_m2"].describe()

# %%
# Results after applying weights
print("RMSE:")
print(
    metrics.root_mean_squared_error(
        weighted_results["msoa_avg_outdoor_space_m2"],
        weighted_results["msoa_weighted_average_garden_area_m2"],
    )
)
print("Median absolute error:")
print(
    metrics.median_absolute_error(
        weighted_results["msoa_avg_outdoor_space_m2"],
        weighted_results["msoa_weighted_average_garden_area_m2"],
    )
)

# %%
fig, ax = plt.subplots()
print(len(weighted_results))
plot = weighted_results.filter(
    pl.col("msoa_avg_outdoor_space_m2") < 1600,
    pl.col("msoa_weighted_average_garden_area_m2") < 4000,
)
print(len(plot))
plt.scatter(
    plot["msoa_avg_outdoor_space_m2"],
    plot["msoa_weighted_average_garden_area_m2"],
    alpha=0.3,
    s=1,
)
plt.xlabel("ONS MSOA average (m2)")
plt.ylabel("Estimated MSOA average (m2)")
plt.plot(
    np.unique(plot["msoa_avg_outdoor_space_m2"]),
    np.poly1d(
        np.polyfit(
            plot["msoa_avg_outdoor_space_m2"],
            plot["msoa_weighted_average_garden_area_m2"],
            1,
        )
    )(np.unique(plot["msoa_avg_outdoor_space_m2"])),
    color="red",
    linestyle="--",
    label="Line of best fit",
)
plt.plot(
    np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
    np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
    color="orange",
    label="Ideal line",
)
plt.legend()
plt.title("ONS MSOA average garden size vs our estimates (excluding outliers)")
