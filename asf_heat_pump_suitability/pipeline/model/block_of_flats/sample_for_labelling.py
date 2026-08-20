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
    from datetime import date
    import simplekml
    import polars as pl
    import boto3
    import os
    from asf_heat_pump_suitability.getters import load_data, load_geodata
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns, local_authority

    seed = 10

    # ------------------------------------ #
    # LOAD GRID SQUARES
    # ------------------------------------ #
    args = parse_arguments()
    local_authorities = [la.lower() for la in args.local_authorities]
    local_authority_dict = local_authority.get_dict_la_data(local_authorities)
    grid_squares = local_authority_dict["grid_squares"]

    # ------------------------------------ #
    # LOAD DOMESTIC UPRNS
    # ------------------------------------ #
    # Load our domestic UPRNs from processing
    domestic_uprns = set(pl.read_parquet("path", columns="UPRN")["UPRN"])

    # Load the lookup with all the additional data
    country_mapping = {
        "E92000001": "England",
        "W92000004": "Wales",
        "S92000003": "Scotland",
    }
    uprns_df = (
        load_data.load_df_uprn_lookup()
        .filter(
            pl.col("UPRN").is_in(domestic_uprns),
        )
        .with_columns(
            pl.col("ctry25cd").replace(country_mapping).alias("country"),
            pl.col("ladcd").str.starts_with("E09").alias("in_london"),
        )
    )

    # ------------------------------------ #
    # IMPUTE FLAT LABELS
    # ------------------------------------ #
    flat_uprns = property_type.impute_set_flat_properties(
        uprns_df, x_col="GRIDGB1E", y_col="GRIDGB1N"
    )
    uprns_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("is_flat"))

    # ------------------------------------ #
    # MAP UPRNS TO BUILDINGS
    # ------------------------------------ #
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

    # ------------------------------------ #
    # ENRICH WITH EPC BUILDING AGE DATA
    # ------------------------------------ #
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

    # ------------------------------------ #
    # AGGREGATE UP TO BUILDING LEVEL
    # ------------------------------------ #
    buildings_df = (
        uprns_df.filter(pl.col("N_flats") > 1)
        .group_by("building_id")
        .agg(
            pl.col("UPRN").n_unique().alias("n_uprns"),
            pl.col("n_flats").max().alias("n_flats"),
            pl.col("construction_age_band").max().alias("construction_age_band"),
            pl.col("country").max().alias("country"),
            pl.col("ruc21ind").max().alias("rurality"),
            pl.col("imd19ind").max().alias("IMD_decile"),
            pl.col("in_london").max().alias("in_london"),
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
            pl.when(pl.col("in_london"))
            .then(pl.lit("London"))
            .otherwise(pl.col("country"))
            .alias("area"),
            # Group age bands
            pl.when(
                pl.col("contruction_age_band").is_in(
                    [
                        "England and Wales: before 1900",
                        "1900-1929",
                        "Scotland: Before 1919",
                    ]
                )
            )
            .then(pl.lit("Before 1929"))
            .when(pl.col("contruction_age_band") == "1930-1949")
            .then(pl.lit("1930-1949"))
            .when(pl.col("contruction_age_band").is_in(["1950-1966", "1966-1975"]))
            .then(pl.lit("1950-1975"))
            .when(
                pl.col("contruction_age_band").is_in(
                    ["1976-1983", "1983-1991", "1991-1998", "1996-2002"]
                )
            )
            .then(pl.lit("1976-2002"))
            .when(pl.col("contruction_age_band").is_in(["2003-2007", "2007 onwards"]))
            .then(pl.lit("2003 onwards")),
        )
    )

    # ------------------------------------ #
    # TAKE SAMPLE
    # ------------------------------------ #

    # ------------------------------------ #
    # ADD GOOGLE MAPS URL TO EACH SAMPLE
    # ------------------------------------ #
    # Convert to 4326 projection and create google maps URL
    sample_gdf = sample_gdf.to_crs(epsg=4326)
    sample_gdf = sample_gdf.merge(
        sample_gdf.centroid.get_coordinates(),
        how="left",
        left_index=True,
        right_index=True,
    )
    sample_gdf["url"] = sample_gdf.apply(
        lambda row: f"https://www.google.com/maps/search/?api=1&query={row['y']},{row['x']}",
        axis=1,
    )

    # ------------------------------------ #
    # SAVE TO KML FILE
    # ------------------------------------ #
    today = date.today().strftime("%Y%m%d")
    s3 = boto3.resource("s3")
    BUCKET = "asf-heat-pump-suitability"
    kml = simplekml.Kml()
    for idx, r in sample_gdf.iterrows():
        pol = kml.newpolygon(
            name="unlabelled",
            description=f"Location: https://www.google.com/maps/search/?api=1&query={r['y']},{r['x']}\nN flats: {r['n_flats']}\nN total: {r['n_total']}",
            outerboundaryis=list(r["geometry"].exterior.coords),
        )
        pol.style.polystyle.color = "9939FF14"
        pol.style.polystyle.outline = 1
    l = len(sample_gdf)
    fpath = (
        f"{today}_UNLABELLED_GB_buildings_containing_flats_sample_n{l}_seed{seed}.kml"
    )
    kml.save(fpath)
    s3.Bucket(BUCKET).upload_file(
        os.path.join(os.getcwd(), fpath),
        os.path.join("local_heat_planning", "labelling", fpath),
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
# 10. [DONE] Enrich sample with additional data:
# [DONE] - URL
# [DONE] - Count of UPRNs per building
# 11. [DONE] Convert to kml file
