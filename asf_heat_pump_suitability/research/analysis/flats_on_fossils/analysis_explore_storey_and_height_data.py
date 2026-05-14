# %% [markdown]
# ## Explore FLAT_STOREY_COUNT and building height data
#
# FLAT_STOREY_COUNT data is available from EPC data and building height estimates are available from Microsoft ML Global Building Footprints dataset.
#
# FLAT_STOREY_COUNT has low coverage of non-null values in EPC and Microsoft ML Global Building Footprints has decent coverage. Here we investigate these data to get a sense of their quality and how we can apply the building height data to fill in the storey count variable.

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.analysis.flats_on_fossils.features import building_rise

# %% [markdown]
# ## 1. Load processed EPC data for flats
# This file is the result of processing the EPC data with the `asf_heat_pump_suitability/analysis/flats_with_fossils/run_process_epc_flats.py` script.

# %%
# Import latest EPC data
flats_epc_path = "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/2024_Q3_epc_flats_processed.parquet"
flats_epc_df = pl.read_parquet(flats_epc_path)

# %%
# This replaces values below 1 or any non-integer values with Nulls
flats_epc_df = building_rise.clean_col_flat_storey_count(flats_epc_df)

# %%
# Add building rise column based on storey count only
flats_epc_df = building_rise.extend_df_building_rise_from_storey_count(
    flats_epc_df,
    storey_col="FLAT_STOREY_COUNT",
    building_rise_col="epc_storey_building_rise",
)

# %%
flats_epc_df.shape

# %%
# Null proportions
flats_epc_df.select(
    [
        "FLAT_STOREY_COUNT",
        "height",
    ]
).null_count() / len(flats_epc_df)

# %% [markdown]
# ## 2. Filter out Scotland flats
# We have found a high number of flats with unreasonable meters per storey values using the EPC flat storey count and Microsoft building height data. Upon investigation, most of these seem to be originating in Scotland.
#
# - There seems to be lots of errors in the EPC storey count for scottish flats
# - We notice this when we calculate meters per storey for buildings from Microsoft building height estimates / EPC flat storey count
# - There are lots of buildings with >5 meters per storey which seems too high even including roof
# - We took two random samples of flats and searched them on google maps to collect the true storey count:
# - Sample of 15 flats with >5mps (all happened to be in Scotland in the sample) - 0/15 had accurate storey count
# - Sample of 15 flats from Scotland showed 3/15 had accurate storey count (20%)
# - Using height data and actual storey count in these samples, we see reasonable meters per storey in most cases. Therefore we conclude the EPC storey count data for Scotland appears to be systemically erroneous but the height data seems (from the random samples) reasonable
# - Results of checking the sample show high levels of inaccuracy of Scottish storey count data
# - We will exclude Scottish flats from the model training data
#
# We looked into potential reasons for why the Scottish flats may have systemic errors. One reason could be that according to online research, the ground floor is not counted as a storey. So what we identify in England as a 2 storey building would be considered one storey in Scotland.
# When we account for this, the accuracy of the Scottish storey count increases but it is still below 50%. Therefore we will still exclude Scottish flats from this analysis.
#
# We considered training the model on England and Wales data and then applying it to Scottish height data to estimate storey count. However, according to team local knowledge, Scotland tends to have higher storey heights than England and Wales. Therefore, it may not be appropriate to apply a model trained on English and Welsh data to Scottish flats.
#
# We also filter out buildings <2m in height because we assume this to be erroneous.
#

# %%
# Filter out Scottish data from our analysis df
analysis_df = flats_epc_df.filter(pl.col("COUNTRY") != "Scotland")

# %% [markdown]
# Below we show the results that led us to remove Scottish data.

# %%
# Filter to rows with required data points
# Calculate meters per storey
model_df = (
    flats_epc_df.filter(
        pl.col("FLAT_STOREY_COUNT").is_not_null(),
        pl.col("height").is_not_null(),
        pl.col("property_per_m2").is_not_null(),
    )
    .with_columns(
        (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
    )
    .filter((pl.col("height") >= 2))  # removing rows with buildings smaller than 2m
)

# %%
# The majority of properties with a meters per storey value above 5m are in Scotland
model_df.filter(pl.col("meters_per_storey") > 5)["COUNTRY"].value_counts()

# %%
# Get random sample of flats with >5m per storey (assumed to be erroneous)
sample_df = flats_epc_df.filter(
    pl.col("FLAT_STOREY_COUNT").is_not_null(),
    pl.col("height").is_not_null(),
    pl.col("property_per_m2").is_not_null(),
).with_columns(
    (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
)

sample_df = (
    sample_df.filter(pl.col("meters_per_storey") > 5)
    .sample(15, seed=1)
    .select(
        ["ADDRESS1", "ADDRESS2", "POSTCODE", "COUNTRY", "FLAT_STOREY_COUNT", "height"]
    )
)

sample_df.write_csv("../../../outputs/data/flats/sample_buildings_erroneous_mps.csv")

# %%
# Get random sample of flats from Scotland
sample_df = flats_epc_df.filter(
    pl.col("FLAT_STOREY_COUNT").is_not_null(),
    pl.col("height").is_not_null(),
    pl.col("property_per_m2").is_not_null(),
    pl.col("COUNTRY") == "Scotland",
)

sample_df = sample_df.sample(15, seed=1).select(
    ["ADDRESS1", "ADDRESS2", "POSTCODE", "COUNTRY", "FLAT_STOREY_COUNT", "height"]
)

sample_df.write_csv("../../../outputs/data/flats/sample_scotland_buildings.csv")

# %%
# Load and view results of analysis by uncommenting below - erroneous mps sample
results_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/sample_buildings_erroneous_mps_results.csv"
)
results_df

# %%
# Scotland sample
results_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/sample_scotland_buildings_results.csv"
)
results_df

# %% [markdown]
# ## 3. View distributions of building height and flat storey count data in analysis df

# %%
analysis_df["height"].describe()

# %%
# Remove building heights of less than 2m
analysis_df = analysis_df.with_columns(
    pl.when(pl.col("height") < 2).then(None).otherwise(pl.col("height")).alias("height")
)

# %%
analysis_df["height"].describe()

# %%
# Check distribution of building height data after removing small height values
height_df = analysis_df.filter(pl.col("height").is_not_null()).select(
    ["UPRN", "FLAT_STOREY_COUNT", "height"]
)

plt.hist(height_df["height"], bins=60)
plt.title(
    f"Distribution of building height (m) of EPC records of flats, N={len(height_df)}"
)
plt.xlabel("Building height (m)")
plt.ylabel("Number of records")
plt.show()

# %%
# See distribution of storey counts for EPC flats
storey_count_df = analysis_df.filter(pl.col("FLAT_STOREY_COUNT").is_not_null()).select(
    ["UPRN", "FLAT_STOREY_COUNT", "height", "epc_storey_building_rise"]
)

plot_df = storey_count_df.group_by("FLAT_STOREY_COUNT").agg(
    count=pl.col("UPRN").count()
)

plt.bar(plot_df["FLAT_STOREY_COUNT"], plot_df["count"])
plt.title(f"Number of EPC flats records per storey count, N={len(storey_count_df)}")
plt.xlabel("Storey count")
plt.ylabel("Number of records")
plt.xlim(0, 30)
plt.show()

# %%
storey_count_df["FLAT_STOREY_COUNT"].describe()

# %%
storey_count_df["epc_storey_building_rise"].value_counts()

# %%
# Coverage of flats with height estimate data
len(analysis_df.filter(pl.col("height").is_not_null())) / len(analysis_df)

# %%
# Coverage of flats with UPRN counts per building data
len(analysis_df.filter(pl.col("UPRN_count_per_building").is_not_null())) / len(
    analysis_df
)

# %%
# Coverage of flats with height estimate or flat storey count data
len(
    analysis_df.filter(
        pl.col("height").is_not_null() | pl.col("FLAT_STOREY_COUNT").is_not_null()
    )
) / len(analysis_df)

# %% [markdown]
# ## 4. Compare FLAT_STOREY_COUNT and building height estimate data
#
# Below we plot the number of storeys against the estimated building heights. The space between the lines on the plot indicate where most points should fall if they have a storey height between 2-4m per storey plus a roof of 1.5m. The plot shows the data are likely not accurate for many records with either flat storey count or height estimates being incorrect.
#
# In the second plot, we show meters per storey by building height. This plot indicates there are many buildings reported to be 1 storey regardless of building height (i.e. where meters per storey equals building height). Again, this suggests there are errors in the data.

# %%
# Set these boundaries to plot our data against
min_storey_height = 2
max_storey_height = 4
min_roof_height = 0  # flat roof
max_roof_height = 3

# %%
# Plot number of storeys against building height estimates
storey_height_df = analysis_df.filter(
    pl.col("FLAT_STOREY_COUNT").is_not_null(), pl.col("height").is_not_null()
).with_columns(
    (pl.col("height") / pl.col("FLAT_STOREY_COUNT")).alias("meters_per_storey")
)

storey_counts = np.arange(1, storey_height_df["FLAT_STOREY_COUNT"].max())
min_building_heights = (storey_counts * min_storey_height) + min_roof_height
max_building_heights = (storey_counts * max_storey_height) + max_roof_height

plt.scatter(
    storey_height_df["FLAT_STOREY_COUNT"],
    storey_height_df["height"],
    alpha=0.05,
    label="EPC flat",
)

plt.plot(
    storey_counts,
    min_building_heights,
    color="yellow",
    label=f"{min_storey_height}m per storey",
)

plt.plot(
    storey_counts,
    max_building_heights,
    color="red",
    linestyle="dashed",
    label=f"{max_storey_height}m per storey + {max_roof_height}m roof",
)

plt.plot(
    storey_counts,
    np.poly1d(
        np.polyfit(storey_height_df["FLAT_STOREY_COUNT"], storey_height_df["height"], 1)
    )(storey_counts),
    color="darkblue",
    linestyle="dotted",
    label="Line of best fit",
)

plt.xlim(1, storey_height_df["FLAT_STOREY_COUNT"].max() + 2)
plt.ylim(0, storey_height_df["height"].max() + 5)

plt.title(f"Number of storeys by building height (m), N={len(storey_height_df)}")
plt.xlabel("Number of storeys")
plt.ylabel("Building height (m)")
plt.legend()
plt.show()

# %%
"""Fits a Linear Model:

Estimates the average height per storey (gradient) using linear regression.
Validates the assumed roof_constant by checking the intercept.
Visualises the Data:

Creates a hexbin plot to show the density of buildings by storey count and height.
Adds reference lines (e.g., 2m/storey, 4m/storey) for comparison.
Overlays the best-fit line from the regression model.
Provides Insights:

Helps understand the relationship between storey count and building height.
Validates assumptions about average storey height and roof height."""

import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
import numpy as np

# Assumed roof height
roof_constant = 1.5  # adjust as needed

# Prepare X and y for regression
X = np.array(storey_height_df["FLAT_STOREY_COUNT"]).reshape(-1, 1)
y = np.array(storey_height_df["height"] - roof_constant)

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
gradient = model.coef_[0]
intercept = model.intercept_

# Print the estimated average storey height
print(f"Estimated average storey height (gradient): {gradient:.2f} m")
print(
    f"Intercept (should be near zero if roof_constant is accurate): {intercept:.2f} m"
)

# Generate best-fit line to overlay
storey_range = np.arange(1, 31)
best_fit_heights = gradient * storey_range + roof_constant

# Plot
plt.figure(figsize=(10, 6), dpi=120)
hb = plt.hexbin(
    x=storey_height_df["FLAT_STOREY_COUNT"],
    y=storey_height_df["height"],
    gridsize=50,
    cmap="Blues",
    mincnt=1,
    norm=matplotlib.colors.LogNorm(),
)
plt.colorbar(hb, label="Count of Buildings")

# Reference lines
plt.plot(storey_counts, min_building_heights, color="yellow", label="2m/storey")
plt.plot(
    storey_counts,
    max_building_heights,
    color="red",
    linestyle="dashed",
    label="4m/storey",
)

# Best-fit line
plt.plot(
    storey_range,
    best_fit_heights,
    color="black",
    linestyle="dashdot",
    linewidth=2,
    label=f"Best fit: {gradient:.2f}m/storey",
)

plt.xlim(1, 30)
plt.ylim(0, 40)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("Number of storeys")
plt.ylabel("Building height (m)")
plt.title(f"Number of storeys by building height (m), N={len(storey_height_df)}")
plt.legend()
plt.show()

# %%
# Plot meters per storey by building height
plt.scatter(
    storey_height_df["height"], storey_height_df["meters_per_storey"], alpha=0.01
)
plt.axhline(y=2, color="yellow", label="2m per storey")
plt.axhline(y=4, color="red", linestyle="--", label="4m per storey")
plt.title(f"Meters per storey by building height (m), N={len(storey_height_df)}")
plt.xlabel("Building height (m)")
plt.ylabel("Meters per storey")
plt.legend()
plt.show()

# %%
# Hexbin meters per storey plot
plt.hexbin(
    storey_height_df["height"],
    storey_height_df["meters_per_storey"],
    gridsize=100,
    cmap="viridis",
    bins="log",
)
plt.colorbar(label="log10(N)")
plt.title(f"Meters per storey by building height (m), N={len(storey_height_df):,}")
plt.xlabel("Building height (m)")
plt.ylabel("Meters per storey")
plt.show()

# %%
# Show distribution of meters per storey with peaks

# Get the meters per storey data
meters_per_storey = storey_height_df["meters_per_storey"].to_numpy()

# Compute histogram values
hist_vals, bin_edges = np.histogram(meters_per_storey, bins=100, density=True)

# Find peaks
peaks, properties = find_peaks(
    hist_vals, height=0.01
)  # tweak height threshold if needed

# Convert bin indices to x-axis positions
peak_positions = bin_edges[peaks]

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(
    meters_per_storey, bins=100, density=True, color="steelblue", edgecolor="white"
)

# Plot mean and median
mean_mps = meters_per_storey.mean()
median_mps = np.median(meters_per_storey)
plt.axvline(
    mean_mps,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {mean_mps:.2f} m",
)
plt.axvline(
    median_mps,
    color="orange",
    linestyle=":",
    linewidth=1.5,
    label=f"Median = {median_mps:.2f} m",
)

# Annotate detected peaks
for pos in peak_positions:
    plt.axvline(
        pos, color="green", linestyle="--", linewidth=1, label=f"Peak at {pos:.2f} m"
    )

# Remove duplicate peak labels in legend
handles, labels = plt.gca().get_legend_handles_labels()
seen = set()
new_handles_labels = [
    (h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))
]
plt.legend(*zip(*new_handles_labels))

# Final plot formatting
plt.xlabel("Meters per storey")
plt.ylabel("Density")
plt.title("Distribution of estimated floor height per storey with peaks")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Building height and property density per building rise
#
# Below, we use the building rise type derived from the EPC flat storey count information to segment our dataset into the different rise types. We then plot the distribution of building heights for each rise type and then the IQR and median for each type.
#
# The IQRs for all the building rise and the distributions look ok but not great with clearly erroneous data in some of the tails / outside the IQRs.
#
# We also look at property density of buildings per building rise (UPRNs per building). Note, that UPRNs per building are considered for the whole building footprint (e.g. an entire row of terraced houses), not individual building sections. This is why we use UPRN per m2 which takes into account the total area of the building footprint as well.

# %%
# See average meters per storey per rise type
temp = storey_height_df.group_by("epc_storey_building_rise").agg(
    pl.col("meters_per_storey").median()
)
dict(zip(temp["epc_storey_building_rise"], temp["meters_per_storey"]))

# %%
# See distribution of building height estimates per building rise type (according to EPC data)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for ax, rise in zip(axs.ravel(), ["low-rise", "medium-rise", "high-rise"]):
    plot_df = storey_height_df.filter(pl.col("epc_storey_building_rise") == rise)
    ax.hist(plot_df["height"], bins=40)
    ax.set_title(rise.title())
    ax.set_xlabel("Building height estimate (m)")
    ax.set_ylabel("Count of flats")
    ax.axvline(
        plot_df["height"].quantile(0.25),
        color="r",
        linestyle="-",
        label=f"Lower quartile {round(plot_df['height'].quantile(0.25), 2)}",
    )
    ax.axvline(
        plot_df["height"].quantile(0.50),
        color="y",
        linestyle="--",
        label=f"Median {round(plot_df['height'].quantile(0.5), 2)}",
    )
    ax.axvline(
        plot_df["height"].quantile(0.75),
        color="r",
        linestyle="-.",
        label=f"Upper quartile {round(plot_df['height'].quantile(0.75), 2)}",
    )
    ax.legend()

plt.suptitle("Distribution of building height estimates by building rise in EPC flats")
plt.tight_layout()
plt.show()

# %%
# See distribution of building height estimates per building rise type (according to EPC data)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

for rise in ["low-rise", "medium-rise", "high-rise"]:
    ax.hist(
        storey_height_df.filter(pl.col("epc_storey_building_rise") == rise)["height"],
        bins=40,
        alpha=0.5,
        label=rise.capitalize(),
        density=True,
    )

plt.suptitle("Distribution of building height estimates by building rise in EPC flats")
plt.xlabel("Building height estimate (m)")
plt.ylabel("Count of flats")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# See distribution of UPRN building count per building rise type (according to EPC data)
uprn_count_df = analysis_df.filter(
    (pl.col("UPRN_count_per_building").is_not_null())
    & (pl.col("epc_storey_building_rise").is_not_null())
)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))

for ax, rise in zip(axs.ravel(), ["low-rise", "medium-rise", "high-rise"]):
    plot_df = uprn_count_df.filter(pl.col("epc_storey_building_rise") == rise)
    ax.boxplot(plot_df["UPRN_count_per_building"])
    ax.set_title(rise.title())
    ax.set_ylabel("UPRN count per building")

plt.suptitle("Distribution of UPRN count per building per building rise in EPC flats")
plt.tight_layout()
plt.show()

# %%
# See distribution of property density per building rise type (according to EPC data)
density_df = analysis_df.filter(
    (pl.col("property_per_m2").is_not_null())
    & (pl.col("epc_storey_building_rise").is_not_null())
)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

for rise in ["low-rise", "medium-rise", "high-rise"]:
    ax.hist(
        density_df.filter(pl.col("epc_storey_building_rise") == rise)[
            "property_per_m2"
        ],
        bins=40,
        alpha=0.5,
        label=rise.capitalize(),
        density=True,
    )

plt.suptitle("Distribution of property density by building rise in EPC flats")
plt.xlabel("Property density (per m2)")
plt.ylabel("Count of flats")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Check a sample of buildings
# Here we create a random sample of buildings - 5 from each building rise type - to manually look up on google maps. We will count the number of visible storeys for each building in the sample.

# %%
# Create full address
storey_height_df = storey_height_df.with_columns(
    (pl.col("ADDRESS1") + " " + pl.col("ADDRESS2") + " " + pl.col("POSTCODE")).alias(
        "full_address"
    )
)

# %%
# Create and save sample dataset
samples = [
    storey_height_df.filter(pl.col("epc_storey_building_rise") == rise_type).sample(
        5, seed=1
    )
    for rise_type in storey_height_df["epc_storey_building_rise"].unique()
]
sample_df = pl.concat(samples)
save_utils.save_to_s3(
    sample_df,
    "s3://asf-heat-pump-suitability/test_data/sample_building_height_and_storey_count.csv",
)

# %% [markdown]
# ## 6. After manual review of sample
# I took a random sample of flats (5 each from low-, medium- and high-rise groups) and looked up the addresses on Google maps outside of this notebook. I counted above-ground storeys by counting the number of windows - labelled `manual_check_max_above_ground_storey_count`. I added a flag called `basement_visible` which indicates whether an assumed livable basement storey is visible from Google maps. The basement is NOT included in my storey count. I also added a `variable_height` flag which indicates whether the building appears to have a variable height (e.g. different sections have different numbers of storeys). The storey count is for the max number of storeys in any visible building segment. I added this flag because it may explain some errors in the building height data which is calculated by taking the average height of the building.

# %%
# Load reviewed sample
sample_df = pl.read_csv(
    "s3://asf-heat-pump-suitability/test_data/sample_building_height_and_storey_count_results.csv"
)

# %%
# Select required columns
sample_df = sample_df.select(
    [
        "FLAT_STOREY_COUNT",
        "height",
        "epc_storey_building_rise",
        "meters_per_storey",
        "manual_check_max_above_ground_storey_count",
        "basement_visible",
        "mixed_height",
    ]
)

# %%
# Accepted meters per storey - this is the min and max meters per storey we accept as valid
# 2.3m is the national standard minimum ceiling height for 75% of gross internal area
# 6m max comes from assuming a 3-4m ceiling with 3-2m roof - this would be for a single storey building
min_mps = 2.3
max_mps = 6

sample_df = sample_df.with_columns(
    # Calculate storey count including basement flats
    (
        pl.col("manual_check_max_above_ground_storey_count")
        + pl.col("basement_visible")
    ).alias("manual_check_max_total_storey_count"),
    # Calculate 'true' meters per storey
    (pl.col("height") / pl.col("manual_check_max_above_ground_storey_count")).alias(
        "true_meters_per_storey"
    ),
).with_columns(
    (pl.col("manual_check_max_total_storey_count") - pl.col("FLAT_STOREY_COUNT")).alias(
        "true_minus_epc"
    ),
    pl.when(
        pl.col("FLAT_STOREY_COUNT") == pl.col("manual_check_max_total_storey_count")
    )
    .then(True)
    .otherwise(False)
    .alias("valid_storey_count"),
    pl.when(
        (pl.col("true_meters_per_storey") >= min_mps)
        & (pl.col("true_meters_per_storey") <= max_mps)
    )
    .then(True)
    .otherwise(False)
    .alias("valid_height"),
)

# %%
sample_df["valid_height"].value_counts()

# %%
sample_df["valid_storey_count"].value_counts()

# %%
sample_df.filter(~pl.col("valid_height"))

# %% [markdown]
# ## Conclusion
#
# In this very small sample, the FLAT_STOREY_COUNT data seems to be more accurate, showing accurate storey count for 12/15 of the sample (80%). For one, it can include basement storeys which I don't think the Microsoft data can do because it's looking at building footprints from satellite images. Conversely, building height estimates seems to be within a reasonable range in 7/15 cases, 47% of the time. A reasonable range is defined as 2.3-6m per storey.
#
# The Microsoft data seems to perform worse on medium- and high-rise buildings and seems to underestimate their height (given the very small meters per storey that we get for these rise types).
#
# I can see that 75% (6/8) of the buildings which it has underestimated height for have a variable height. This will reduce the overall average height and thus could be responsible for the errors.
#
# For this reason, I will proceed with the assumption that the FLAT_STOREY_COUNT data is the 'ground truth'. I think it's best to proceed with a model with a variable meters per storey to convert building height data to storey count - i.e. the taller the building, the lower the meters per storey will need to be to accurately convert height to storey counts.
