"""
Create a sample of buildings containing flats for manual labelling to use in model training.
"""

import argparse
import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def load_df_scotland_postcode_lookup() -> pl.DataFrame:
    """
    Load Scotland postcode lookup containing Scottish postcodes and their corresponding population data.

    Returns:
        pl.DataFrame: Scotland postcode lookup
    """
    df = pl.read_csv(config["data"]["lookups"]["scotland"])
    return df.with_columns(
        pl.col("Postcode").str.strip_chars().name.keep(),
    ).select(["Postcode", "DataZone2011Code"])


def load_df_lsoa_imd_decile(nation: str = None) -> pl.DataFrame:
    """
    Load IMD decile for LSOAs (England & Wales) or Data Zones (Scotland) in GB.

    Args:
        nation (str): nation to load IMD decile data for, of "England", "Scotland" or "Wales". Default None to load all nations.

    Returns:
        pl.DataFrame: IMD decile data per LSOA / Data Zone for specified nation.
    """
    dfs = []
    # England
    if not nation or nation.lower() == "england":
        df = pl.read_csv(config["data"]["imd_deciles"]["england"])
        dfs.append(
            df.rename(
                {
                    "LSOA code (2021)": "LSOA_or_DZ",
                    "Index of Multiple Deprivation (IMD) Decile (where 1 is most deprived 10% of LSOAs)": "IMD_decile",
                }
            ).select(["LSOA_or_DZ", "IMD_decile"])
        )

    # Scotland
    if not nation or nation.lower() == "scotland":
        df = base_getters.get_df_from_excel_s3_path(
            config["data"]["imd_deciles"]["scotland"],
            sheet_name="SIMD 2020v2 DZ lookup data",
        )
        dfs.append(
            df.rename({"DZ": "LSOA_or_DZ", "SIMD2020v2_Decile": "IMD_decile"}).select(
                ["LSOA_or_DZ", "IMD_decile"]
            )
        )

    # Wales
    if not nation or nation.lower() == "wales":
        df = pl.read_csv(config["data"]["imd_deciles"]["wales"])
        dfs.append(
            df.filter(
                pl.col("Domain") == "WIMD", pl.col("Data description") == "Decile"
            )
            .rename({"Area code": "LSOA_or_DZ", "Data values": "IMD_decile"})
            .select(["LSOA_or_DZ", "IMD_decile"])
        )

    return pl.concat(dfs)


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
        default=["GB"],
        required=False,
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the input UPRN file to access. Defaults to today's date.",
        required=False,
    )

    parser.add_argument(
        "--seed",
        help="Seed for random sampling. Default 7.",
        default=7,
        required=False,
    )

    parser.add_argument(
        "--target_n",
        help="Target size of sample. Default 3000.",
        default=3000,
        required=False,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    from datetime import date
    import simplekml
    import boto3
    import os
    import math
    from asf_heat_pump_suitability.getters import load_data, load_geodata
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns, local_authority
    from asf_heat_pump_suitability.utils import save_utils

    # ------------------------------------ #
    # LOAD ARGS
    # ------------------------------------ #
    args = parse_arguments()
    release_date = save_utils.get_str_release_date(args.release_date)
    seed = args.seed
    target_n = args.target_n

    # ------------------------------------ #
    # LOAD GRID SQUARES
    # ------------------------------------ #
    local_authorities = [la.lower() for la in args.local_authorities]
    local_authority_dict = local_authority.get_dict_la_data(local_authorities)
    grid_squares = local_authority_dict["grid_squares"]

    # ------------------------------------ #
    # LOAD DOMESTIC UPRNS
    # ------------------------------------ #
    # Load our domestic UPRNs from processing
    print("Load domestic UPRNs and NSUL...")
    slug = local_authority_dict["url_slug"]
    fpath = config["output"]["dataset"]["domestic_uprns"].format(
        local_authority=slug, release_date=release_date
    )
    domestic_uprns = pl.scan_parquet(fpath).collect().select(["UPRN"])

    # Load the lookup with all the additional data
    uprns_df = (
        load_data.load_df_uprn_lookup(
            uprn_filter=domestic_uprns,
            columns=[
                "UPRN",
                "ctry25cd",
                "lad25cd",
                "PCDS",
                "lsoa21cd",
                "GRIDGB1E",
                "GRIDGB1N",
                "ruc21ind",
            ],
        )
        .with_columns(
            pl.col("lad25cd").str.starts_with("E09").alias("in_london"),
        )
        .rename({"ctry25cd": "country"})
    )
    del domestic_uprns

    # ------------------------------------ #
    # ADD IMD DECILE DATA
    # ------------------------------------ #
    print("Add IMD decile data...")
    scotland_dz_lookup_df = load_df_scotland_postcode_lookup()
    imd_df = load_df_lsoa_imd_decile()

    uprns_df = (
        uprns_df.join(
            scotland_dz_lookup_df, how="left", left_on="PCDS", right_on="Postcode"
        )
        .with_columns(
            pl.when(pl.col("country") == "S92000003")
            .then(pl.col("DataZone2011Code"))
            .otherwise(pl.col("lsoa21cd"))
            .alias("IMD_LSOA_or_DZ")
        )
        .join(imd_df, how="left", left_on="IMD_LSOA_or_DZ", right_on="LSOA_or_DZ")
    )
    del scotland_dz_lookup_df, imd_df

    # ------------------------------------ #
    # IMPUTE FLAT LABELS
    # ------------------------------------ #
    print("Impute flat labels...")
    flat_uprns = property_type.impute_set_flat_properties(
        uprns_df, x_col="GRIDGB1E", y_col="GRIDGB1N"
    )
    uprns_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("is_flat"))
    del flat_uprns

    # ------------------------------------ #
    # MAP UPRNS TO BUILDINGS
    # ------------------------------------ #
    print("Map UPRNs to building footprints...")
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )
    uprns_gdf = uprns.generate_gdf_uprn_coords(
        uprns_df,
        usecols=["UPRN", "GRIDGB1E", "GRIDGB1N"],
        x_col="GRIDGB1E",
        y_col="GRIDGB1N",
    )
    uprn_building_mapping = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf, id_col="ID"
    )
    uprns_df = uprns_df.with_columns(
        # Map building IDs to the UPRNs they contain
        pl.col("UPRN")
        .replace_strict(uprn_building_mapping, default=None)
        .alias("building_id")
    ).with_columns(pl.col("is_flat").sum().over("building_id").alias("n_flats"))
    del uprns_gdf, uprn_building_mapping

    # ------------------------------------ #
    # ENRICH WITH EPC BUILDING AGE DATA
    # ------------------------------------ #
    print("Enrich with EPC building age data...")
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
            .otherwise(pl.col("construction_age_band"))
            .name.keep()
        )
    )
    uprns_df = uprns_df.join(epc_df, how="left", on="UPRN")
    del epc_df

    # ------------------------------------ #
    # AGGREGATE UP TO BUILDING LEVEL
    # ------------------------------------ #
    print("Aggregate to building level...")
    buildings_df = (
        uprns_df.filter(pl.col("n_flats") > 1)
        .group_by("building_id")
        .agg(
            pl.col("UPRN").n_unique().alias("n_uprns"),
            pl.col("n_flats").max().alias("n_flats"),
            pl.col("construction_age_band").max().alias("construction_age_band"),
            pl.col("country").max().alias("country"),
            pl.col("ruc21ind").max().alias("rurality"),
            pl.col("IMD_decile").max().alias("IMD_decile"),
            pl.col("in_london").max().alias("in_london"),
        )
        .with_columns(
            # Add proportion of flats
            (pl.col("n_flats") / pl.col("n_uprns")).alias("proportion_flats"),
            # Group flat count
            pl.when(pl.col("n_flats").is_between(2, 6, closed="both"))
            .then(pl.lit("2-6_flats"))
            .when(pl.col("n_flats").is_between(7, 15, closed="both"))
            .then(pl.lit("7-15_flats"))
            .when(pl.col("n_flats") > 15)
            .then(pl.lit("16+_flats"))
            .otherwise(None)
            .alias("n_flats_grouped"),
            # Add London as a separate area
            pl.when(pl.col("in_london"))
            .then(pl.lit("London"))
            .otherwise(pl.col("country"))
            .alias("area"),
            # Group construction age bands
            pl.when(
                pl.col("construction_age_band").is_in(
                    [
                        "England and Wales: before 1900",
                        "1900-1929",
                        "Scotland: Before 1919",
                    ]
                )
            )
            .then(pl.lit("Before 1929"))
            .when(pl.col("construction_age_band") == "1930-1949")
            .then(pl.lit("1930-1949"))
            .when(pl.col("construction_age_band").is_in(["1950-1966", "1966-1975"]))
            .then(pl.lit("1950-1975"))
            .when(
                pl.col("construction_age_band").is_in(
                    ["1976-1983", "1983-1991", "1991-1998", "1996-2002"]
                )
            )
            .then(pl.lit("1976-2002"))
            .when(pl.col("construction_age_band").is_in(["2003-2007", "2007 onwards"]))
            .then(pl.lit("2003 onwards"))
            .otherwise(None)
            .alias("grouped_construction_age_band"),
            # Group IMD deciles
            pl.when(pl.col("IMD_decile").is_in([1, 2, 3]))
            .then(pl.lit("high_deprivation"))
            .when(pl.col("IMD_decile").is_in([4, 5, 6, 7]))
            .then(pl.lit("middle_deprivation"))
            .when(pl.col("IMD_decile").is_in([8, 9, 10]))
            .then(pl.lit("low_deprivation"))
            .otherwise(None)
            .alias("deprivation_group"),
            # Group rurality
            pl.when(pl.col("rurality").is_in(["UN1", "UF1", "1", "2"]))
            .then(pl.lit("urban"))
            .when(pl.col("rurality").is_in(["RLN1", "RLF1", "3", "4"]))
            .then(pl.lit("large_rural"))
            .when(pl.col("rurality").is_in(["RSN1", "RSF1", "5", "6"]))
            .then(pl.lit("small_rural"))
            .otherwise(None)
            .alias("rurality"),
        )
        .with_columns((pl.col("proportion_flats") > 0.8).alias("over_80_pc_flats"))
    )

    del uprns_df

    save_utils.save_to_s3(
        df=buildings_df,
        path="s3://asf-local-heat-planning-tool/outputs/models/block_of_flats_classifier/gb_enriched_buildings_with_flats.parquet",
    )

    # ------------------------------------ #
    # TAKE SAMPLE
    # ------------------------------------ #
    print("Take sample of buildings...")
    attributes = [
        "area",
        "rurality",
        "grouped_construction_age_band",
        "n_flats_grouped",
        "deprivation_group",
        "over_80_pc_flats",
    ]

    buildings_df = buildings_df.with_columns(
        pl.col(attributes).cast(pl.String).fill_null("unknown")
    )
    # Get the number of combinations of attributes, which is the number of groups to sample from
    n_combinations = buildings_df.group_by(attributes).agg(pl.len()).height
    sample_n = target_n // n_combinations
    print(
        f"There are {n_combinations} groups to sample from. Taking {sample_n} samples from each group."
    )

    # Sample per group
    sampled_ids = (
        buildings_df.group_by(attributes)
        .agg(
            # Sample building IDs from each group
            pl.col("building_id").sample(n=sample_n, with_replacement=False, seed=seed)
        )
        .explode("building_id")
        .select("building_id")
    )

    # Filter population dataset to sample IDs
    sample_df = buildings_df.filter(pl.col("building_id").is_in(sampled_ids))
    del buildings_df
    sample_gdf = buildings_gdf[["ID", "geometry"]].merge(
        sample_df.to_pandas(), how="inner", left_on="ID", right_on="building_id"
    )
    del buildings_gdf

    # ------------------------------------ #
    # ADD GOOGLE MAPS URL TO EACH SAMPLE
    # ------------------------------------ #
    print("Enrich with Google Maps URL...")
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
    print("Save to KML file...")
    today = date.today().strftime("%Y%m%d")
    s3 = boto3.resource("s3")
    BUCKET = "asf-heat-pump-suitability"
    kml = simplekml.Kml()
    for idx, r in sample_gdf.iterrows():
        pol = kml.newpolygon(
            name="unlabelled",
            description=f"Location: {r['url']}\nN flats: {r['n_flats']}\nN total: {r['n_total']}",
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
