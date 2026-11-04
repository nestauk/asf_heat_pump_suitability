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
from typing import defaultdict
from asf_heat_pump_suitability.utils import s3_utils
from asf_heat_pump_suitability.pipeline.transform import local_authority

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
    for la in tqdm(la_names):
        # Load LA contextual features dataset
        slug = local_authority.make_str_slug(la)
        output_gdf = gpd.read_file(
            f"s3://asf-local-heat-planning-tool/outputs/data/{slug}/{date}/{slug}_clusters_contextual_features_5m.geojson"
        )
        # Remove non-cluster layers
        output_gdf = output_gdf[
            output_gdf["layer"] == "clusters_with_contextual_features"
        ]
        # Collate numeric columns
        cols = ["assigned_tech"] + [
            col
            for col in output_gdf.select_dtypes(include="number").columns
            if not col.startswith("perc_")
        ]
        # Calculate sum of each numeric feature per assigned tech
        summary_df = output_gdf[cols].groupby("assigned_tech").agg("sum").reset_index()
        summary_df[["local_authority", "nation"]] = la, nation_name
        results.append(summary_df)

results_df = pl.from_pandas(pd.concat(results))


# %%
def calculate_pc_x_per_group(
    df: pl.DataFrame, x: str, group: str, by_nation: bool = False
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
        agg_df = (
            df.with_columns(pl.col(x).sum().over("nation").alias(f"{x}_in_nation"))
            .group_by(["nation", group])
            .agg(pl.col(x).sum(), pl.col(f"{x}_in_nation").first())
            .with_columns(
                (pl.col(x) / pl.col(f"{x}_in_nation") * 100).alias(f"pc_{x}_in_nation")
            )
            .sort(by=["nation", group])
        )
    else:
        agg_df = df
        agg_df = (
            df.with_columns(pl.col(x).sum().alias(f"total_{x}"))
            .group_by(group)
            .agg(pl.col(x).sum(), pl.col(f"total_{x}").first())
            .with_columns((pl.col(x) / pl.col(f"total_{x}") * 100).alias(f"pc_{x}"))
            .sort(by="nation")
        )

    # Show full numbers without scientific notation and with commas
    with pl.Config(
        set_fmt_float="full",
        thousands_separator=True,
    ):
        print(agg_df)

    return agg_df


# %%
# What is the number of homes across GB suitable for heat pumps (individual solutions)?
# For all of GB
calculate_pc_x_per_group(df=results_df, x="n_UPRNs", group="assigned_tech")

# %%
# What is the number of homes across GB suitable for heat pumps (individual solutions)?
# Per nation
by_nation_df = calculate_pc_x_per_group(
    df=results_df, x="n_UPRNs", group="assigned_tech", by_nation=True
)

with pl.Config(
    set_fmt_float="full",
    thousands_separator=True,
):
    print(by_nation_df.filter(pl.col("assigned_tech") == "Individual solution"))

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

with pl.Config(
    set_fmt_float="full",
    thousands_separator=True,
):
    print(by_nation_df.filter(pl.col("assigned_tech") == "Individual solution"))

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
