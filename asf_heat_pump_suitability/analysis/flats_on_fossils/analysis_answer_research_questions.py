# %% [markdown]
# ## Creating national-level statistics on flats with fossil fuels
#
# In this notebook, we conduct our final analysis on the processed EPC flats data to answer the original research questions.
#
# Research questions:
# 1. How many flats are there in the UK?
# 2. What is the proportion of each flat type (e.g. high rise, low rise)?
# 3. What proportion of flats use communal heating in the UK? (& heat networks?)
# 4. What is the breakdown of fuel types of this communal heating (e.g. electric, gas etc)?
# 5. What proportion of different flat types (low, medium and high rise/ number of floors of whole building + any other details we have e.g. maisonette) use gas/ other fossil fuels - communally (as opposed to directly electric, communal heating powered by electricity)?
# 6. What proportion of flats use individual heating in the UK?
# 7. What proportion of different flat types (low, medium and high rise/ number of floors of whole building + any other details we have e.g. maisonette) use gas/ other fossil fuels - individually?
# 8. What flat types are most commonly heated individually with gas?

# %%
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import ticker
from asf_heat_pump_suitability.getters import get_target

# %% [markdown]
# ## 1. Load processed EPC data for flats
# This file is the result of processing the EPC data with the `asf_heat_pump_suitability/analysis/flats_with_fossils/run_process_epc_flats.py` script and filling in the missing storey count data with the `asf_heat_pump_suitability/analysis/flats_with_fossils/rfill_building_rise_methods.ipynb` notebook.

# %%
# Import latest EPC data
flats_epc_path = "s3://asf-heat-pump-suitability/outputs/2024Q3/analysis/2024_Q3_epc_flats_processed_filled_storey_count.parquet"
raw_flats_epc_df = pl.read_parquet(flats_epc_path)

# %%
flats_epc_df = raw_flats_epc_df

# %% [markdown]
# ## Null counts
#
# Here we check the null counts of our input values.
# All but 2 of the missing `fuel_type` values are in flats with community heating.

# %%
# Null counts
flats_epc_df.select(
    [
        "PROPERTY_TYPE",
        "building_rise",
        "fuel_type",
        "community_heating",
        "fossil_fuel_heating",
    ]
).null_count()

# %%
# Null proportions
flats_epc_df.select(
    [
        "PROPERTY_TYPE",
        "building_rise",
        "fuel_type",
        "community_heating",
        "fossil_fuel_heating",
    ]
).null_count() / len(flats_epc_df)

# %%
flats_epc_df.filter(pl.col("fuel_type").is_null())["community_heating"].value_counts()

# %%
flats_epc_df.filter(pl.col("fuel_type").is_null())[
    "MAINHEAT_DESCRIPTION"
].value_counts()

# %% [markdown]
# ## Preprocessing
# Here is an additional preprocessing step to rename the building rise categories for plotting

# %%
rise_order = [
    "Low-rise\n(1-3 storeys)",
    "Medium-rise\n(4-10 storeys)",
    "High-rise\n(11+ storeys)",
    "Unknown",
]

building_rise_dict = {
    "low-rise": "Low-rise\n(1-3 storeys)",
    "medium-rise": "Medium-rise\n(4-10 storeys)",
    "high-rise": "High-rise\n(11+ storeys)",
}

flats_epc_df = raw_flats_epc_df.with_columns(
    pl.col("building_rise")
    .replace_strict(building_rise_dict, default="Unknown")
    .alias("building_rise_title")
).with_columns(
    pl.col("building_rise_title").cast(pl.Enum(rise_order)),
)

# %% [markdown]
# ## Answer research questions

# %% [markdown]
# ## 1. How many flats are there in England, Scotland, and Wales?
#
# Information from combined Scotland and England and Wales census data: 6,238,634.

# %%
census_property_df = get_target.transform_df_target_property_type()
census_property_df["Flat, maisonette or apartment"].sum()

# %%
len(flats_epc_df)

# %% [markdown]
# ## 2. What is the proportion of each flat type (e.g. high rise, low rise)?

# %%
total_buildings_per_rise_df = (
    flats_epc_df["building_rise_title"]
    .value_counts(sort=True, normalize=False)
    .rename({"count": "total_count"})
)
total_buildings_per_rise_df

# %%
plot_df = (
    flats_epc_df["building_rise_title"]
    .value_counts(normalize=True)
    .sort(by="building_rise_title")
)
plot_df

# %%
plot_df = (
    flats_epc_df.filter(pl.col("building_rise").is_not_null())["building_rise_title"]
    .value_counts(normalize=True)
    .sort(by="building_rise_title")
)
plot_df

# %%
labels = list(
    zip(plot_df["building_rise_title"].to_list(), (plot_df["proportion"] * 100).round())
)
labels = [k + f", {int(v)}%" for k, v in labels]

plt.pie(
    plot_df["proportion"],
    labels=labels,
    startangle=90,
    counterclock=False,
    colors=["#0000FF", "#FDB633", "#18A48C"],
    wedgeprops={"edgecolor": "black", "linewidth": 1},
)
plt.title("Proportion of EPC flats by building rise type in GB")
plt.tight_layout()
plt.show()

# %%
# Including null values
plot_df = (
    flats_epc_df["building_rise_title"]
    .value_counts(normalize=True)
    .sort(by="building_rise_title")
)
plt.pie(
    plot_df["proportion"],
    labels=plot_df["building_rise_title"],
    startangle=90,
    counterclock=False,
    colors=["#0000FF", "#FDB633", "#9A1BBE", "#18A48C"],
    wedgeprops={"edgecolor": "black", "linewidth": 1},
)
plt.title("Proportion of EPC flats by building rise type in GB")
plt.show()

# %% [markdown]
# ## 3. What proportion of flats use communal/ individual heating in the UK? (& heat networks?)

# %%
flats_epc_df["community_heating"].value_counts(sort=True, normalize=False)

# %%
flats_epc_df["community_heating"].value_counts(sort=True, normalize=True)

# %%
plot_df = (
    flats_epc_df.filter(pl.col("building_rise").is_not_null())
    .group_by(["building_rise_title", "community_heating"])
    .agg(pl.col("UPRN").count())
    .join(total_buildings_per_rise_df, how="left", on="building_rise_title")
    .with_columns(
        (pl.col("UPRN") / pl.col("total_count") * 100).alias("percentage_per_rise_type")
    )
)
plot_df

# %%
plot_df = plot_df.filter(pl.col("community_heating")).sort(by="building_rise_title")
plt.bar(plot_df["building_rise_title"], plot_df["percentage_per_rise_type"])
plt.ylim(0, 50)
plt.title("Percentage of EPC flats with community heating by building rise type")
plt.xlabel("Building rise type")
plt.ylabel("Percentage of flats with community heating")
plt.tight_layout()
plt.show()

# %%
# Comparison with EHS data
flats_epc_df = flats_epc_df.with_columns(
    pl.when(pl.col("govt_defined_high_rise").is_null())
    .then(None)
    .when(pl.col("govt_defined_high_rise"))
    .then(pl.lit("high-rise"))
    .otherwise(pl.lit("low-rise"))
    .alias("ehs_building_rise")
)

total_ehs_buildings_per_rise_df = (
    flats_epc_df["ehs_building_rise"]
    .value_counts(sort=True, normalize=False)
    .rename({"count": "total_count"})
)

flats_epc_df.filter(pl.col("ehs_building_rise").is_not_null()).group_by(
    ["ehs_building_rise", "community_heating"]
).agg(pl.col("UPRN").count()).join(
    total_ehs_buildings_per_rise_df, how="left", on="ehs_building_rise"
).with_columns(
    (pl.col("UPRN") / pl.col("total_count") * 100).alias("percentage_per_rise_type")
)

# %%
# Proportion of high-rise flats with communal heating according to EHS
print(211 / 616)

# %%
# Proportion of non high-rise flats with communal heating according to EHS
print(298 / 4780)

# %% [markdown]
# ## 4. What is the breakdown of fuel types of this communal heating (e.g. electric, gas etc)?
# Community heating seems to be where most of the fuel type data is missing. 44% of rows are missing fuel type information here.

# %%
flats_epc_df.filter(pl.col("community_heating"))["fuel_type"].value_counts(
    sort=True, normalize=False
)

# %%
flats_epc_df.filter(pl.col("community_heating"))["fuel_type"].value_counts(
    sort=True, normalize=True
)

# Save as csv to get full table of results
# flats_epc_df.filter(pl.col("community_heating"))["fuel_type"].value_counts(sort=True, normalize=True).with_columns((pl.col("proportion")*100).round(3).alias("percentage")).write_csv("community_heating.csv")

# %%
flats_epc_df.filter(pl.col("community_heating"))["fossil_fuel_heating"].value_counts(
    sort=True, normalize=False
)

# %%
flats_epc_df.filter(pl.col("community_heating"))["fossil_fuel_heating"].value_counts(
    sort=True, normalize=True
)

# %% [markdown]
# ## 5. What proportion of different flat types (low, medium and high rise/ number of floors of whole building + any other details we have e.g. maisonette) use gas/ other fossil fuels - communally (as opposed to direct electric, communal heating powered by electricity)?

# %%
total_buildings_per_rise_community_df = (
    flats_epc_df.filter(pl.col("community_heating"))["building_rise_title"]
    .value_counts(sort=True, normalize=False)
    .rename({"count": "total_count"})
)
total_buildings_per_rise_community_df

# %%
plot_df = (
    flats_epc_df.filter(
        pl.col("building_rise").is_not_null(), pl.col("community_heating")
    )
    .group_by(["building_rise_title", "fossil_fuel_heating"])
    .agg(pl.col("UPRN").count())
    .join(total_buildings_per_rise_community_df, how="left", on="building_rise_title")
    .with_columns(
        (pl.col("UPRN") / pl.col("total_count") * 100).alias("percentage_per_rise_type")
    )
    .sort(by="building_rise_title")
)
plot_df

# %%
plot_df = (
    plot_df.pivot(
        on="fossil_fuel_heating",
        index="building_rise_title",
        values="percentage_per_rise_type",
    )
    .select(["building_rise_title", "true", "false", "null"])
    .rename(
        {
            "building_rise_title": "Building rise type",
            "true": "Fossil fuel heating",
            "false": "Renewable heating",
            "null": "Unknown",
        }
    )
    .to_pandas()
)
plot_df.index = plot_df["Building rise type"]
plot_df = plot_df.drop("Building rise type", axis=1)

plot_df.plot(kind="bar")

plt.ylabel("Percentage of flats")
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
plt.gca().xaxis.set_tick_params(rotation=0)
plt.title("Percentage of communally heated flats on different heating fuel types")
plt.show()

# %% [markdown]
# ## 6. What proportion of different flat types (low, medium and high rise/ number of floors of whole building + any other details we have e.g. maisonette) use gas/ other fossil fuels - individually?
#

# %%
total_buildings_per_rise_individual_df = (
    flats_epc_df.filter(~pl.col("community_heating"))["building_rise_title"]
    .value_counts(sort=True, normalize=False)
    .rename({"count": "total_count"})
)
total_buildings_per_rise_individual_df

# %%
plot_df = (
    flats_epc_df.filter(
        pl.col("building_rise").is_not_null(), ~pl.col("community_heating")
    )
    .group_by(["building_rise_title", "fossil_fuel_heating"])
    .agg(pl.col("UPRN").count())
    .join(total_buildings_per_rise_individual_df, how="left", on="building_rise_title")
    .with_columns(
        (pl.col("UPRN") / pl.col("total_count") * 100).alias("percentage_per_rise_type")
    )
    .sort(by="building_rise_title")
)
plot_df

# %%
plot_df = (
    plot_df.pivot(
        on="fossil_fuel_heating",
        index="building_rise_title",
        values="percentage_per_rise_type",
    )
    .select(["building_rise_title", "true", "false"])
    .rename(
        {
            "building_rise_title": "Building rise type",
            "true": "Fossil fuel heating",
            "false": "Renewable heating",
        }
    )
    .to_pandas()
)
plot_df.index = plot_df["Building rise type"]
plot_df = plot_df.drop("Building rise type", axis=1)

plot_df.plot(kind="bar")

plt.ylabel("Percentage of flats")
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
plt.gca().xaxis.set_tick_params(rotation=0)
plt.title("Percentage of individually heated flats on different heating fuel types")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. What flat types are most commonly heated individually with gas?

# %%
flats_epc_df.filter(
    pl.col("building_rise").is_not_null(),
    ~pl.col("community_heating"),
    pl.col("fuel_type") == "gas",
).group_by(["PROPERTY_TYPE", "building_rise_title"]).agg(
    pl.col("UPRN").count()
).with_columns(
    (pl.col("UPRN") / pl.col("UPRN").sum() * 100).alias("percentage")
).sort(
    by="percentage", descending=True
)

# %% [markdown]
# ## 8. Heating type/percentage of each heating type by single floor, 2 floor and 3 floor blocks

# %%
total_flats_per_storey_count_df = (
    flats_epc_df.filter(pl.col("storey_count") <= 30)["storey_count"]
    .value_counts(sort=True, normalize=False)
    .rename({"count": "total_count"})
)
total_flats_per_storey_count_df

# %%
plot_df = (
    flats_epc_df.filter(pl.col("storey_count") <= 30)
    .group_by(["storey_count", "fossil_fuel_heating"])
    .agg(pl.col("UPRN").count())
    .join(total_flats_per_storey_count_df, how="left", on="storey_count")
    .with_columns(
        pl.col("storey_count").cast(pl.Int64).alias("storey_count"),
        (pl.col("UPRN") / pl.col("total_count") * 100).alias(
            "percentage_per_storey_count"
        ),
    )
    .sort(by="storey_count")
)
plot_df

# %%
plot_df = (
    plot_df.pivot(
        on="fossil_fuel_heating",
        index="storey_count",
        values="percentage_per_storey_count",
    )
    .select(["storey_count", "true", "false", "null"])
    .rename(
        {
            "storey_count": "Number of storeys",
            "true": "Fossil fuel heating",
            "false": "Renewable heating",
            "null": "Unknown heating type",
        }
    )
    .to_pandas()
)
plot_df.index = plot_df["Number of storeys"]
plot_df = plot_df.drop("Number of storeys", axis=1)

plot_df.plot(kind="bar")

plt.ylabel("Percentage of flats")
plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
plt.gca().xaxis.set_tick_params(rotation=0)
plt.title("Percentage of flats on different heating fuel types by number of storeys")
plt.tight_layout()
plt.show()

# %%
plt.scatter(plot_df.index, plot_df["Fossil fuel heating"])
plt.plot(plot_df.index, plot_df["Fossil fuel heating"], label="Fossil fuel heating")
plt.scatter(plot_df.index, plot_df["Renewable heating"])
plt.plot(plot_df.index, plot_df["Renewable heating"], label="Renewable heating")
plt.scatter(plot_df.index, plot_df["Unknown heating type"])
plt.plot(plot_df.index, plot_df["Unknown heating type"], label="Unknown heating type")
plt.title("Percentage of flats on different heating fuel types by number of storeys")
plt.xlabel("Number of storeys")
plt.ylabel("Percentage of flats")
plt.legend()
plt.show()
