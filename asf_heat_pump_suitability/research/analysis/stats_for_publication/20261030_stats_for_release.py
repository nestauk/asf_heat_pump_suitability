# %% [markdown]
# ## Calculate statistics of interest for publication
#
# To complement the release of the tool, we are interesting in publishing some high-level statistics from our local heat planning dataset. This notebook calculates the numbers to answer the following questions:
#
# - What is the number of homes across GB suitable for heat pumps (individual solutions)?
# - What is the % of homes in social housing that could be suitable for heat pumps (individual solutions)?

# %%
from tqdm import tqdm
import pandas as pd
import polars as pl
import geopandas as gpd
from asf_heat_pump_suitability.getters import load_data
from asf_heat_pump_suitability.pipeline.transform import local_authority, uprns
from asf_heat_pump_suitability.pipeline.model.block_of_flats import sample_for_labelling

# %%
# Rural urban classification
ru_lookup_df = load_data.load_df_uprn_lookup(
    columns=[
        "UPRN",
        "PCDS",
        "ruc21ind",
    ],
)

scotland_ru_lookup_df = sample_for_labelling.load_df_scotland_postcode_lookup()

ru_lookup_df = (
    (
        ru_lookup_df.join(
            scotland_ru_lookup_df, how="left", left_on="PCDS", right_on="Postcode"
        )
    )
    .with_columns(
        pl.when(pl.col("ruc21ind").is_in(["UN1", "UF1", "1", "2"]))
        .then(pl.lit("Urban"))
        .when(
            pl.col("ruc21ind").is_in(
                ["RLN1", "RLF1", "3", "4", "RSN1", "RSF1", "5", "6"]
            )
        )
        .then(pl.lit("Rural"))
        .otherwise(None)
        .alias("urban_rural")
    )
    .select(["UPRN", "urban_rural"])
)

# %%
date = "20261030"

nations = {"E": "England", "S": "Scotland", "W": "Wales"}

results = []

for nation_code, nation_name in nations.items():
    print(f"\n\nProcessing LAs for {nation_name}")
    # Load LAs for the nation
    la_names = local_authority.load_list_la_names(nation=nation_code)
    # Ensure names are consistent with file names
    resolved_las = local_authority.resolve_list_la_names(la_names)
    for la in tqdm(resolved_las):
        slug = local_authority.make_str_slug(la)

        # Load UPRNs dataset
        uprns_df = pl.read_parquet(
            f"s3://asf-local-heat-planning-tool/outputs/data/{slug}/{date}/{slug}_domestic_uprns.parquet"
        ).join(ru_lookup_df.select(["UPRN", "urban_rural"]), how="left", on="UPRN")
        uprns_gdf = uprns.generate_gdf_uprn_coords(
            df=uprns_df, usecols=["UPRN", "urban_rural"]
        ).to_crs(epsg=27700)[["UPRN", "urban_rural", "geometry"]]

        # Load LA contextual features dataset
        output_gdf = gpd.read_file(
            f"s3://asf-local-heat-planning-tool/outputs/data/{slug}/{date}/{slug}_clusters_contextual_features_5m.geojson"
        ).to_crs(epsg=27700)
        # Remove non-cluster layers
        output_gdf = output_gdf[
            output_gdf["layer"] == "clusters_with_contextual_features"
        ]

        # Map urban / rural label to cluster
        cluster_to_uprn_gdf = output_gdf.sjoin(
            uprns_gdf, how="left", predicate="contains"
        ).drop(columns="index_right")
        ruc_mapping_df = (
            pl.from_pandas(cluster_to_uprn_gdf.drop(columns="geometry"))
            .group_by("cluster_id")
            .agg(
                pl.col("urban_rural").max().alias("urban_rural_max"),
                pl.col("urban_rural").n_unique().alias("n_unique"),
            )
            .with_columns(
                pl.when(pl.col("n_unique") > 2)
                .then(pl.lit("mixed"))
                .otherwise(pl.col("urban_rural_max"))
                .alias("urban_rural")
            )
            .select(["cluster_id", "urban_rural"])
            .to_pandas()
        )

        output_gdf = output_gdf.merge(ruc_mapping_df, how="left", on="cluster_id")

        # Collate numeric columns
        cols = ["assigned_tech", "urban_rural"] + [
            col
            for col in output_gdf.select_dtypes(include="number").columns
            if not col.startswith("perc_")
        ]
        # Calculate sum of each numeric feature per assigned tech
        summary_df = (
            output_gdf[cols]
            .groupby(["assigned_tech", "urban_rural"])
            .agg("sum")
            .reset_index()
        )
        summary_df[["local_authority", "nation"]] = la, nation_name
        results.append(summary_df)

results_df = pl.from_pandas(pd.concat(results))


# %%
def calculate_pc_x_per_group(
    df: pl.DataFrame, x: str, group: str | list, by_nation: bool = False
) -> pl.DataFrame:
    """
    Calculate the percentage of a feature's (x) total that falls into each group, and print it.

    For example, with x="n_UPRNs" and group="assigned_tech", this gives the number and
    percentage of UPRNs assigned to each technology.

    Args:
        df (pl.DataFrame): local authority counts of numerical features, containing columns `x` and `group` (and "nation" if
        `by_nation` is True).
        x (str): numeric column to sum and take percentages of.
        group (str): categorical column to group by, e.g. "assigned_tech". `x` is summed within each value of this column (e.g.
        each technology) and reported as a percentage of the total.
        by_nation (bool): if `True`, percentages are relative to each nation's total rather than the GB total. Default `False`
        to calculate percentages at GB level.

    Returns:
        pl.DataFrame: one row per group (per nation and group if `by_nation`), with the summed `x`, the total, and the percentage of
        either the GB or by-nation total
    """
    if by_nation:
        if isinstance(group, str):
            group = [group]
        agg_df = (
            df.with_columns(pl.col(x).sum().over("nation").alias(f"{x}_in_nation"))
            .group_by(["nation"] + group)
            .agg(pl.col(x).sum(), pl.col(f"{x}_in_nation").first())
            .with_columns(
                (pl.col(x) / pl.col(f"{x}_in_nation") * 100).alias(f"pc_{x}_in_nation")
            )
            .sort(by=["nation"] + group)
        )
    else:
        agg_df = (
            df.with_columns(pl.col(x).sum().alias(f"total_{x}"))
            .group_by(group)
            .agg(pl.col(x).sum(), pl.col(f"total_{x}").first())
            .with_columns((pl.col(x) / pl.col(f"total_{x}") * 100).alias(f"pc_{x}"))
            .sort(by=group)
        )

    # Show full numbers without scientific notation and with commas
    with pl.Config(
        set_fmt_float="full",
        thousands_separator=True,
    ):
        print(agg_df)

    return agg_df


# %%
def print_individual_solutions(df: pl.DataFrame) -> pl.DataFrame:
    with pl.Config(
        set_fmt_float="full",
        thousands_separator=True,
    ):
        print(df.filter(pl.col("assigned_tech") == "Individual solution"))


# %%
# What is the number of homes across GB suitable for heat pumps (individual solutions)?
# For all of GB
calculate_pc_x_per_group(df=results_df, x="n_UPRNs", group="assigned_tech")
calculate_pc_x_per_group(
    df=results_df, x="n_UPRNs", group=["assigned_tech", "urban_rural"]
)

# %%
# What is the number of homes across GB suitable for heat pumps (individual solutions)?
# Per nation
by_nation_df = calculate_pc_x_per_group(
    df=results_df, x="n_UPRNs", group="assigned_tech", by_nation=True
)

ruc_by_nation_df = calculate_pc_x_per_group(
    df=results_df, x="n_UPRNs", group=["assigned_tech", "urban_rural"], by_nation=True
)

print_individual_solutions(by_nation_df)
print_individual_solutions(ruc_by_nation_df)

# %%
# What is the % of homes in social housing that could be suitable for heat pumps (individual solutions)?
# For all of GB
calculate_pc_x_per_group(
    df=results_df, x="tenure_rental_(social)", group="assigned_tech"
)

# %%
# What is the % of homes in social housing that could be suitable for heat pumps (individual solutions)?
# Per nation
by_nation_df = calculate_pc_x_per_group(
    df=results_df, x="tenure_rental_(social)", group="assigned_tech", by_nation=True
)

ruc_by_nation_df = calculate_pc_x_per_group(
    df=results_df,
    x="tenure_rental_(social)",
    group=["assigned_tech", "urban_rural"],
    by_nation=True,
)

print_individual_solutions(by_nation_df)
print_individual_solutions(ruc_by_nation_df)

# %%
# What is the % of homes with unknown tenure?
# For all of GB
total_uprns = results_df["n_UPRNs"].sum()
total_unknown = results_df["tenure_unknown"].sum()
print(
    f"Percentage of properties in GB with unknown tenure: {total_unknown / total_uprns * 100}"
)

# %%
# What is the % of homes with unknown tenure?
# Per nation
agg_df = (
    results_df.with_columns(
        pl.col("n_UPRNs").sum().over("nation").alias("n_UPRNs_per_nation")
    )
    .group_by("nation")
    .agg(
        pl.col("tenure_unknown").sum().alias("n_tenure_unknown"),
        pl.col("n_UPRNs_per_nation").first(),
    )
    .with_columns(
        (pl.col("n_tenure_unknown") / pl.col("n_UPRNs_per_nation") * 100).alias(
            "pc_tenure_unknown"
        )
    )
)

with pl.Config(
    set_fmt_float="full",
    thousands_separator=True,
):
    print(agg_df)
