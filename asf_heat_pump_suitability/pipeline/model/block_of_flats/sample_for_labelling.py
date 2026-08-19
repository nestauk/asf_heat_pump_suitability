"""
Create a sample of buildings containing flats for manual labelling to use in model training.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_authorities",
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        default="GB",
        required=False,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if "name" == "__main__":
    import polars as pl
    from asf_heat_pump_suitability.getters import load_data, load_geodata
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns, local_authority

    args = parse_arguments()

    local_authorities = [la.lower() for la in args.local_authorities]
    local_authority_dict = local_authority.get_dict_la_data(local_authorities)
    grid_squares = local_authority_dict["grid_squares"]

    # Load our domestic UPRNs from processing
    domestic_uprns = set(pl.read_parquet("path", columns="UPRN")["UPRN"])

    # Load the lookup with all the additional data
    uprns_df = load_data.load_df_uprn_lookup().filter(
        pl.col("UPRN").is_in(domestic_uprns)
    )
    flat_uprns = property_type.impute_set_flat_properties(
        uprns_df, x_col="GRIDGB1E", y_col="GRIDGB1N"
    )
    uprns_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("is_flat"))

    # Load building footprints and map to UPRNs
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df, usecols=["UPRN", "is_flat"])
    uprn_building_mapping = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf, id_col="ID"
    )
    uprns_df = uprns_df.with_columns(
        # Map building IDs to the UPRNs they contain
        pl.col("UPRN")
        .replace_strict(uprn_building_mapping, default=None)
        .alias("building_id")
    ).with_columns(pl.col("is_flat").sum().over("building_id").alias("n_flats"))

    epc_df = (
        load_data.load_df_domestic_epc(
            grid_squares=grid_squares, columns=["UPRN", "CONSTRUCTION_AGE_BAND"]
        )
        .with_columns(
            pl.col("CONSTRUCTION_AGE_BAND")
            .str.to_lowercase()
            .str.replace("unknown", "")
            .alias("construction_age_band")
        )
        .with_columns(
            pl.when(pl.col("construction_age_band") == "")
            .then(None)
            .otherwise(pl.col(pl.String))
            .name.keep()
        )
    )
    uprns_df = uprns_df.join(epc_df, how="left", on="UPRN")

    buildings_df = (
        uprns_df.filter(pl.col("N_flats") > 1)
        .group_by("building_id")
        .agg(
            pl.col("UPRN").n_unique().alias("n_uprns"),
            pl.col("n_flats").max().alias("n_flats"),
            pl.col("construction_age_band").max().alias("construction_age_band"),
            pl.col("ctry25cd").max().alias("country"),
            pl.col("ruc21ind").max().alias("rurality"),
            pl.col("imd19ind").max().alias("IMD_decile"),
        )
        .with_columns(
            pl.when(pl.col("n_flats").is_between(2, 6, closed="both"))
            .then(pl.lit("2-6_flats"))
            .when(pl.col("n_flats").is_between(7, 15, closed="both"))
            .then(pl.lit("7-15_flats"))
            .when(pl.col("n_flats") > 15)
            .then(pl.lit("16+_flats"))
            .otherwise(None)
            .alias("n_flats_grouped"),
        )
    )


# STEP-BY-STEP APPROACH
# 1. [DONE] Label all UPRNs as flats or not.
# 2. [DONE] Identify all buildings containing flats.
# 3. [DONE] Label buildings with rural/urban indicator
# 4. [DONE] Label buildings with IMD decile
# 5. [DONE] Label buildings with country
# 7. [DONE] Get count of flats per building
# 8. [DONE] Label buildings with grouped-count
# 9. Sample based on rural/urban indicator; IMD decile; country; and grouped count
# 10. Enrich sample with additional data:
# - URL
# - Count of UPRNs per building
# 11. Convert to kml file
