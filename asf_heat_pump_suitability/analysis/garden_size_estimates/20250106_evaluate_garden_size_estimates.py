# %%
import polars as pl
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
from sklearn import metrics
from asf_heat_pump_suitability.getters import base_getters
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.pipeline.prepare_features import (
    garden_space_avg,
    household_count,
)

# %%
# PARAMS
year = 2023
q = 4
interim_dir = "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/interim/"
nations = "EWS"

primary_colour = "#0000FF"
secondary_colour = "#FF6E47"
tertiary_colour = "#18A48C"

# %% [markdown]
# ## Compare calculated garden size to avg
#
# ### 1. Load average garden per MSOA

# %%
avg_gardens_df = garden_space_avg.generate_df_garden_space_avg()

# %% [markdown]
# ### 2. Load garden size estimates and deduplicate
#
# For properties that get joined to the same garden (i.e. they share a garden, or at least outdoor space) we decided in a sprint review that we would assign all properties which share a garden the overall size of the garden. E.g. if 10 houses share a 100m2 garden, they will all be assigned 100m2 garden size. The logic was that if a building has the appropriate outdoor space, we consider that an apartment within the building has appropriate space in theory (we are also unable to determine which apartment the garden belongs to).
#
# For this analysis, we have to conduct this extra division step which partitions the garden size equally among households who share a garden. E.g. the 10 houses sharing a 100m2 garden will now each get assigned 10m2. This is required to avoid double counting gardens when calculating the average and to make the resulting garden size estimates comparable to the ONS MSOA-level averages which are calculated by counting shared gardens once in their averages.
#
# A caveat of our deduplication here is that if there are apartments from the same building missing from the EPC dataset or missing from our garden size estimates (e.g. because they are missing valid UPRN or there are data gaps in the Microsoft buildings / INSPIRE land parcel datasets), we will overestimate the divided garden size. E.g. if we only have records for 5 of the 10 houses sharing the 100m2 garden in our dataset, they would get assigned 20m2 each instead of 10m2.

# %%
# We load the interim garden size files again and deduplicate them
interim_files = base_getters.list_obj_s3_location(interim_dir)

logging.info("Deduplicating UPRNs that were matched to multiple gardens")
gardens_df = pl.DataFrame()
for file in interim_files:
    logging.info(f"Loading file: {file}")
    df = pl.read_parquet(f"s3://{file}")
    df = df.with_columns(pl.col(pl.Float64).round(2))
    # We cannot take the median garden size because we have to assign each UPRN to a Cadastral Reference for the
    # division step below
    # Therefore we assign each duplicated UPRN the minimum garden size for that UPRN
    df = df.filter(garden_area_m2=pl.col("garden_area_m2").min().over("UPRN"))
    gardens_df = pl.concat([gardens_df, df])

# Final round of deduplication
gardens_df = gardens_df.filter(
    garden_area_m2=pl.col("garden_area_m2").min().over("UPRN")
)

# Save if desired
# save_as = f"s3://asf-heat-pump-suitability/outputs/{year}Q{q}/gardens/{datetime.today().strftime('%Y%m%d')}_{year}_Q{q}_EPC_garden_size_estimates_{nations.upper()}_deduplicated_FOR_EVALUATION_ONLY.parquet"
# save_utils.save_to_s3(gardens_df, save_as)

# %%
gardens_df = pl.read_parquet(
    "s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/20250204_2023_Q4_EPC_garden_size_estimates_EWS_deduplicated_FOR_EVALUATION_ONLY.parquet",
    columns=["UPRN", "garden_area_m2", "NATIONALCADASTRALREFERENCE"],
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

# %%
# Look at garden size estimate distribution
plot_df = gardens_df.unique(subset="NATIONALCADASTRALREFERENCE")
max = plot_df["cadastral_garden_size_m2"].quantile(0.97)

font = {"fontname": "Helvetica"}

fig, ax = plt.subplots()
counts, edges, bars = ax.hist(
    plot_df.filter(pl.col("cadastral_garden_size_m2") <= max)[
        "cadastral_garden_size_m2"
    ],
    bins=50,
    color="#0000FF",
)
ax.set_title(
    "Distribution of estimated garden area for EPC properties\n(where available) in Great Britain (up to 97th percentile)",
    **font,
)
ax.set_xlabel("Estimated garden area (m2)", **font)

scale_y = 1e6
ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale_y))
ax.yaxis.set_major_formatter(ticks_y)
ax.set_ylabel("Count of gardens (millions)", **font)
ax.set_ylim(0, 2_500_000)

plt.tight_layout()
plt.show()

# %%
# Check for disclosiveness of plot
counts

# %%
# Divide shared gardens equally among UPRNs sharing gardens
gardens_df = gardens_df.with_columns(
    (pl.col("cadastral_garden_size_m2") / pl.col("UPRN_count")).alias(
        "divided_garden_area_m2"
    )
)

# %%
len(gardens_df.filter(pl.col("garden_area_m2") <= 10)) / len(gardens_df) * 100

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
        color=primary_colour,
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
    color=primary_colour,
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
    color=secondary_colour,
    linestyle="--",
    label="Line of best fit",
)
plt.plot(
    np.arange(0, 2000), np.arange(0, 2000), color=tertiary_colour, label="Ideal line"
)
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
        color=primary_colour,
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
        color=secondary_colour,
        linestyle="--",
        label="Line of best fit",
    )
    axes[i].plot(
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        color=tertiary_colour,
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
    avg_gardens_df,
    how="left",
    left_on=["msoa", "msoa_avg_outdoor_space_property_type"],
    right_on=["MSOA code", "msoa_avg_outdoor_space_property_type"],
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
weighted_results = epc_df.group_by(
    ["lsoa", "msoa_avg_outdoor_space_property_type"]
).agg(
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
weighted_results = weighted_results.group_by(
    ["msoa", "msoa_avg_outdoor_space_property_type"]
).agg(
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
# Results after applying weights
for property_type in ["Houses", "Flats", "unknown"]:
    print(property_type)
    _df = weighted_results.filter(
        pl.col("msoa_avg_outdoor_space_property_type") == property_type
    )
    print(f"N={len(_df)}")
    print("RMSE:")
    print(
        metrics.root_mean_squared_error(
            _df["msoa_avg_outdoor_space_m2"],
            _df["msoa_weighted_average_garden_area_m2"],
        )
    )
    print("Median absolute error:")
    print(
        metrics.median_absolute_error(
            _df["msoa_avg_outdoor_space_m2"],
            _df["msoa_weighted_average_garden_area_m2"],
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
    color=primary_colour,
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
    color=secondary_colour,
    linestyle="--",
    label="Line of best fit",
)
plt.plot(
    np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
    np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
    color=tertiary_colour,
    label="Ideal line",
)
plt.legend()
plt.title("ONS MSOA average garden size vs our estimates (excluding outliers)")

# %%
# Plot scatter of average garden size estimate vs ONS values per property type
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for i, property_type in enumerate(["Houses", "Flats", "unknown"]):
    plot = weighted_results.filter(
        pl.col("msoa_avg_outdoor_space_property_type") == property_type
    )
    print(len(plot))
    plot = plot.filter(
        pl.col("msoa_avg_outdoor_space_m2") < 12000,
        pl.col("msoa_weighted_average_garden_area_m2") < 4000,
    )
    print(len(plot))
    axes[i].scatter(
        plot["msoa_avg_outdoor_space_m2"],
        plot["msoa_weighted_average_garden_area_m2"],
        alpha=0.3,
        color=primary_colour,
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
                plot["msoa_weighted_average_garden_area_m2"],
                1,
            )
        )(np.unique(plot["msoa_avg_outdoor_space_m2"])),
        color=secondary_colour,
        linestyle="--",
        label="Line of best fit",
    )
    axes[i].plot(
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        np.arange(0, plot["msoa_avg_outdoor_space_m2"].max()),
        color=tertiary_colour,
        label="Ideal line",
    )

plt.gca().legend(("MSOA", "Line of best fit", "Ideal line"))
fig.suptitle("ONS MSOA average garden size vs our estimates (excluding outliers)")
plt.show()

# %%
