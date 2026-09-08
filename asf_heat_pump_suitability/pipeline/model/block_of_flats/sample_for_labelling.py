"""
Create a disproportional stratified sample of buildings containing flats for manual labelling to use in block of flats
classifier model training.

Buildings are grouped by area, rurality, construction age band, number of flats, deprivation
group, and whether the building is predominantly flats (>80%). A target number of samples is
distributed as evenly as possible across these groups. Where a group has fewer buildings than
the per-group target, the remainder is redistributed evenly across groups with remaining
capacity. The sample is enriched with Google Maps URLs and saved as a KML file to S3
for labelling.

Usage:
    python sample_for_labelling.py [options]

Options:
    --local_authorities     One or more local authority names to filter to.
                            Defaults to all of GB.
                            e.g. --local_authorities plymouth
                                 --local_authorities "glasgow city" "south lanarkshire"
    --release_date          Release date of the input UPRN file in YYYYMMDD format.
                            Defaults to today's date.
    --seed                  Random seed for reproducibility. Default: 7.
    --target_n              Target number of buildings in the sample. Default: 3000.
    -m, --map_uprns_to_building
                            If set, regenerates the UPRN-to-building-ID mapping from scratch
                            and saves it to S3. Otherwise loads the existing mapping.
    --save                  If set, saves intermediate outputs to S3.

Examples:
    # Run for all of GB with defaults
    python sample_for_labelling.py

    # Run for Plymouth with a smaller sample
    python sample_for_labelling.py --local_authorities plymouth --target_n 500

    # Run for multiple local authorities and save outputs
    python sample_for_labelling.py --local_authorities "glasgow city" "south lanarkshire" --save

    # Regenerate UPRN to building ID mapping and save
    python sample_for_labelling.py --map_uprns_to_building --save
"""

from typing import List
import argparse
import polars as pl
from scipy.optimize import minimize

import numpy as np
from numpy.typing import ArrayLike

from asf_heat_pump_suitability import config, PROJECT_DIR
from asf_heat_pump_suitability.getters import base_getters

BUILDING_CONSTRUCTION_AGE_MAPPING = {
    "england and wales: before 1900": "1_England and Wales: before 1900",
    "scotland: before 1919": "2_Scotland: Before 1919",
    "1900-1929": "3_1900-1929",
    "1930-1949": "4_1930-1949",
    "1950-1966": "5_1950-1966",
    "1965-1975": "6_1965-1975",
    "1976-1983": "7_1976-1983",
    "1983-1991": "8_1983-1991",
    "1991-1998": "9_1991-1998",
    "1996-2002": "10_1996-2002",
    "2003-2007": "11_2003-2007",
    "2007 onwards": "12_2007 onwards",
    "unknown": "",
}


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


def calculate_int_ssd(
    sample_allocations: np.array,
    primary_group_ids: ArrayLike,
    target_per_primary: float,
) -> float:
    """
    Calculate the sum of squared deviations from the target sample size per primary strata.

    Args:
        sample_allocations (np.array): proposed count of samples to take from each cell
        primary_group_ids (ArrayLike): unique IDs of the primary strata combinations to group by
        target_per_primary (int): target sample number per primary stratum

    Returns:
        int: sum of squared deviations
    """
    # Sum sample allocations for each of the primary strata.
    # This is effectively a vectorised 'groupby' - we group the full combination of sampling cells into their primary
    # strata
    primary_totals = np.bincount(primary_group_ids, weights=sample_allocations)
    # Minimize sum of squared deviations from the target sample size
    return np.sum((primary_totals - target_per_primary) ** 2)


def calculate_array_sample_allocations(
    grouped_df: pl.DataFrame,
    total_sample: int,
    secondary_attributes: List[str],
    primary_col: str = "primary_strata",
) -> np.array:
    """
    Calculate optimal count of samples per group to fulfil primary stratification (proportional or even) and secondary
    constraints.

    Args:
        grouped_df (pl.DataFrame): sampling cells containing building counts for each unique combination of primary and
        secondary attributes.
        total_sample (int): desired sample size for whole sample.
        secondary_attributes (List[str]): list of secondary attributes which will act as constraints in sampling. Must
        be boolean attributes.
        primary_col (str): name of column to be created containing unique IDs for the primary strata combinations.
        Default `primary_strata`.

    Returns:
        np.array: count of samples per group in `grouped_df` required to meet constraints.
    """
    # Total number of cells to optimise sample count from.
    # This is the number of combinations multiplied by the number of groups in each secondary constraint.
    n_cells = grouped_df.height

    # Get the target number of samples per group
    n_primary_groups = grouped_df[primary_col].n_unique()
    target_per_primary = total_sample / n_primary_groups

    # Sample size per cell must be between 0 and real population count
    bounds_per_group = [(0, i) for i in grouped_df["n_buildings"]]

    # Seed the optimiser with an initial guess: evenly distribute sample across all cells
    x0 = np.full(n_cells, total_sample / n_cells)
    objective_args = {
        "primary_group_ids": grouped_df[primary_col],
        "target_per_primary": target_per_primary,
    }

    # Set secondary sampling constraints
    sampling_constraints = [
        # Constraint 1: overall total must equal total_sample requested
        {"type": "eq", "fun": lambda x: np.sum(x) - total_sample},
    ]

    for attr in secondary_attributes:
        # Additional constraints: binary constraints should be distributed 50/50
        sampling_constraints.append(
            {
                "type": "eq",
                "fun": lambda x, _vals=grouped_df[attr].to_numpy().astype(int): np.sum(
                    x * _vals
                )
                - (0.5 * total_sample),
            }
        )

    # Run optimiser to calculate the optimal number of samples from each group
    result = minimize(
        # Objective function to be minimised (i.e. here we are minimising the sum of squared deviations to
        # penalise large deviations from the desired sample size for each group)
        fun=lambda x: calculate_int_ssd(x, **objective_args),
        x0=x0,
        method="SLSQP",
        bounds=bounds_per_group,
        constraints=sampling_constraints,
    )

    # Extract results and round them (they are floats originally)
    sample_allocations = np.round(result.x).astype(int)
    print("Optimized sub-cell sample sizes:", sample_allocations)
    print("Total sampled:", np.sum(sample_allocations))
    for attr in secondary_attributes:
        print(
            f"{attr} total sampled:",
            sum(sample_allocations * grouped_df[attr].to_numpy().astype(int)),
        )

    return sample_allocations


def generate_df_sampling_cells(
    buildings_df: pl.DataFrame,
    primary_attributes: List[str],
    secondary_attributes: List[str],
    group_id_col: str = "group_id",
    primary_col: str = "primary_strata",
) -> pl.DataFrame:
    """
    Generate a DataFrame of cells to sample from based on primary strata and secondary constraints. The resulting
    dataframe will have N rows, where N is equal to the number of categories in each attribute (primary or secondary)
    multiplied together. The cells will contain the counts of buildings in each unique group combination.

    Args:
        buildings_df (pl.DataFrame): buildings where each row represents a unique building, and each row is enriched with
        the specified primary and secondary attributes.
        primary_attributes (List[str]): list of primary attributes to stratify sample by
        secondary_attributes (List[str]): list of secondary attributes which will act as constraints in sampling
        group_id_col (str): name of unique ID column to be created containing sample cell IDs (i.e. unique combinations of primary and
        secondary attributes). Default `group_id`.
        primary_col (str): name of column to be created containing unique IDs for the primary strata combinations. Default `primary_strata`.

    Returns:
        pl.DataFrame: sampling cells containing building counts for each unique combination of primary and secondary attributes
    """
    all_attributes = primary_attributes + secondary_attributes

    return (
        buildings_df.group_by(all_attributes)
        .agg(
            pl.count("building_id").alias("n_buildings"),
        )
        .with_columns(
            pl.concat_list(all_attributes).alias(group_id_col),
            # Create unique ID from primary attributes
            pl.concat_list(primary_attributes)
            .list.join("_")
            .cast(pl.Categorical)
            .to_physical()  # required to cast categorical strings to numbers
            .alias(primary_col),
        )
    )


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
        "-m",
        "--map_uprns_to_building",
        help="Set to generate UPRN to building ID mapping. Otherwise, existing mapping loaded.",
        required=False,
        action="store_true",
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

    print("Map UPRNs to building footprints...")
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=grid_squares
    )

    # Note: putting this at the top of the script prevents memory overload. The datasets loaded in this `if` statement
    # can be very large. Therefore, we load and process them first, then delete them before moving on to the rest of the
    # script.
    if args.map_uprns_to_building:
        print("Generating UPRN to building ID mapping...")
        all_uprns_df = load_geodata.load_df_osopen_uprn(grid_squares=grid_squares)
        all_uprns_gdf = uprns.generate_gdf_uprn_coords(all_uprns_df)

        uprn_building_mapping = uprns.map_dict_uprns_to_building_id(
            uprns_gdf=all_uprns_gdf, buildings_gdf=buildings_gdf, id_col="ID"
        )

        del all_uprns_gdf, all_uprns_df
        # Save mapping
        fpath = config["data"]["processed"]["uprn_to_building_id_mapping"].format(
            local_authorities=local_authorities
        )
        pl.DataFrame(
            {
                "UPRN": uprn_building_mapping.keys(),
                "building_ID": uprn_building_mapping.values(),
            }
        ).write_parquet(fpath)

    # ------------------------------------ #
    # LOAD DOMESTIC UPRNS
    # ------------------------------------ #
    # Load our domestic UPRNs from processing
    print("Load domestic UPRNs and NSUL...")
    slug = local_authority_dict["url_slug"]
    fpath = config["output"]["dataset"]["domestic_uprns"].format(
        local_authority=slug, release_date=release_date
    )
    domestic_uprns = (
        pl.scan_parquet(fpath)
        .collect()
        .select(["UPRN"])
        .with_columns(pl.lit(True).alias("is_domestic"))
    )

    # Load the lookup with all the additional data
    uprns_df = (
        load_data.load_df_uprn_lookup(
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
        .join(domestic_uprns, how="left", on="UPRN")
        .with_columns(pl.col("is_domestic").fill_null(False))
    )
    del domestic_uprns

    # ------------------------------------ #
    # ADD IMD DECILE DATA
    # ------------------------------------ #
    print("Add IMD decile data...")
    # Note: a separate Scotland lookup is required here because: the uprns_df contains 2022 Scottish Data Zones for each
    # UPRN, but the IMD decile data for Scotland is for 2019 DZs. This additional lookup maps postcodes to 2019 DZs, so
    # is ultimately used to map UPRNs to their 2019 DZs, which can then be mapped to their IMD decile.
    # This is required because DZs changed between 2019 and 2022.
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
    uprns_df = uprns_df.with_columns(
        (pl.col("UPRN").is_in(flat_uprns) & pl.col("is_domestic")).alias("is_flat")
    )
    del flat_uprns

    # ------------------------------------ #
    # MAP UPRNS TO BUILDINGS
    # ------------------------------------ #
    if not args.map_uprns_to_building:
        print("Loading UPRN to building ID mapping...")
        uprn_building_mapping = pl.read_parquet(
            "s3://asf-local-heat-planning-tool/outputs/models/block_of_flats_classifier/gb_uprn_to_building_mapping_non_domestic_and_domestic.parquet"
        )
        uprn_building_mapping = dict(
            zip(uprn_building_mapping["UPRN"], uprn_building_mapping["building_ID"])
        )

    uprns_df = (
        uprns_df.with_columns(
            # Map building IDs to the UPRNs they contain
            pl.col("UPRN")
            .replace_strict(uprn_building_mapping, default=None)
            .alias("building_id")
        )
        .with_columns(
            pl.col("is_flat").sum().over("building_id").alias("n_flats"),
            pl.col("is_domestic").sum().over("building_id").alias("n_domestic"),
        )
        .filter(pl.col("n_flats") > 1, pl.col("n_domestic") > 0)
    )
    del uprn_building_mapping

    # ------------------------------------ #
    # ENRICH WITH EPC BUILDING AGE DATA
    # ------------------------------------ #
    print("Enrich with EPC building age data...")
    epc_df = (
        load_data.load_df_domestic_epc(
            grid_squares=grid_squares, columns=["UPRN", "CONSTRUCTION_AGE_BAND"]
        ).with_columns(
            pl.col("CONSTRUCTION_AGE_BAND")
            .str.to_lowercase()
            .replace(BUILDING_CONSTRUCTION_AGE_MAPPING)
            .alias("construction_age_band")
        )
        # This is required to replace empty string with None which we need to do so that the agg max() below works
        # i.e. this prevents 'unknown' from being the max value and ensures consistency in treatment of Null values.
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
        uprns_df.group_by("building_id")
        .agg(
            pl.col("UPRN").n_unique().alias("n_uprns"),
            pl.col("is_domestic").sum().alias("n_domestic_uprns"),
            # The features below are not expected to have different values within the same buildings.
            # Construction age band may vary across different UPRNs within buildings due to either errors in EPC, or
            # due to genuine differences within a larger building footprint. For this reason, we take the earliest value.
            pl.col("n_flats").first().alias("n_flats"),
            pl.col("construction_age_band").min().alias("construction_age_band"),
            pl.col("country").first().alias("country"),
            pl.col("ruc21ind").first().alias("rurality"),
            pl.col("IMD_decile").first().alias("IMD_decile"),
            pl.col("in_london").first().alias("in_london"),
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
                        "1_England and Wales: before 1900",
                        "2_Scotland: Before 1919",
                        "3_1900-1929",
                    ]
                )
            )
            .then(pl.lit("Before 1929"))
            .when(pl.col("construction_age_band") == "4_1930-1949")
            .then(pl.lit("1930-1949"))
            .when(pl.col("construction_age_band").is_in(["5_1950-1966", "6_1965-1975"]))
            .then(pl.lit("1950-1975"))
            .when(
                pl.col("construction_age_band").is_in(
                    ["7_1976-1983", "8_1983-1991", "9_1991-1998", "10_1996-2002"]
                )
            )
            .then(pl.lit("1976-2002"))
            .when(
                pl.col("construction_age_band").is_in(
                    ["11_2003-2007", "12_2007 onwards"]
                )
            )
            .then(pl.lit("2003 onwards"))
            .otherwise(None)
            .alias("grouped_construction_age_band"),
            # Group IMD deciles
            pl.when(pl.col("IMD_decile").is_in([1, 2, 3]))
            .then(True)
            .when(pl.col("IMD_decile").is_in([4, 5, 6, 7, 8, 9, 10]))
            .then(False)
            .otherwise(None)
            .alias("high_deprivation"),
            # Group rurality
            pl.when(pl.col("rurality").is_in(["UN1", "UF1", "1", "2"]))
            .then(True)
            .when(
                pl.col("rurality").is_in(
                    ["RLN1", "RLF1", "3", "4", "RSN1", "RSF1", "5", "6"]
                )
            )
            .then(False)
            .otherwise(None)
            .alias("is_urban"),
        )
        .with_columns(
            (pl.col("proportion_flats") > 0.8)
            .cast(pl.Float64)
            .alias("over_80_pc_flats")
        )
    )

    del uprns_df

    save_utils.save_to_s3(
        df=buildings_df,
        path="s3://asf-local-heat-planning-tool/outputs/models/block_of_flats_classifier/gb_enriched_buildings_with_flats.parquet",
    )

    # ------------------------------------ #
    # TAKE SAMPLE
    # ------------------------------------ #
    buildings_df = pl.read_parquet(
        "s3://asf-local-heat-planning-tool/outputs/models/block_of_flats_classifier/gb_enriched_buildings_with_flats.parquet"
    )
    print("Take sample of buildings...")
    primary_strata = [
        "area",
        "n_flats_grouped",
        "high_deprivation",
    ]

    secondary_constraints = [
        "is_urban",
        "over_80_pc_flats",
    ]

    attributes = primary_strata + secondary_constraints

    buildings_df = buildings_df.with_columns(
        pl.col("high_deprivation").cast(pl.String).fill_null("unknown"),
    )

    secondary_constraints = ["is_urban", "over_80_pc_flats"]

    training_n = round(target_n * 0.75)

    # Create dataframe of cells to calculate sample sizes from
    group_counts_df = generate_df_sampling_cells(
        buildings_df=buildings_df,
        primary_attributes=primary_strata,
        secondary_attributes=secondary_constraints,
    )
    n_samples_per_group = calculate_array_sample_allocations(
        grouped_df=group_counts_df,
        total_sample=training_n,
        secondary_attributes=secondary_constraints,
    )
    group_counts_df = group_counts_df.with_columns(
        pl.Series(name="n_to_sample", values=n_samples_per_group)
    )

    # Sample per group using per-group sampling quota
    buildings_with_quota = buildings_df.join(
        group_counts_df.select(attributes + ["n_to_sample"]),
        on=attributes,
        how="left",
    )

    # Sample IDs randomly from each combination of attributes
    sampled_ids = (
        buildings_with_quota.with_columns(
            # Generates a sequential integer index for all rows in the dataframe from 0 to N-1
            pl.int_range(pl.len())
            # Randomly shuffles the row indices independently within each combination of attributes
            .shuffle(seed=seed).over(attributes)
            # Assigns the randomised integer index to a temporary '_rank' column
            .alias("_rank")
        )
        # Keep rows where '_rank' is less than 'n_to_sample' - this is the method to identify the sample rows
        .filter(pl.col("_rank") < pl.col("n_to_sample")).select("building_id")
    )

    # Filter population dataset to sample IDs
    sample_df = buildings_df.filter(
        pl.col("building_id").is_in(sampled_ids["building_id"].to_list())
    )
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
    sample_gdf[["x", "y"]] = sample_gdf.centroid.get_coordinates()
    sample_gdf["url"] = (
        "https://www.google.com/maps/search/?api=1&query="
        + sample_gdf["y"].astype(str)
        + ","
        + sample_gdf["x"].astype(str)
    )

    # ------------------------------------ #
    # SAVE TO KML FILE
    # ------------------------------------ #
    print("Save to KML file...")
    today = date.today().strftime("%Y%m%d")
    s3 = boto3.resource("s3")
    BUCKET = "asf-local-heat-planning-tool"
    kml = simplekml.Kml()
    for url, n_flats, n_total, geom in zip(
        sample_gdf["url"],
        sample_gdf["n_flats"],
        sample_gdf["n_uprns"],
        sample_gdf["geometry"],
    ):
        pol = kml.newpolygon(
            name="unlabelled",
            description=f"Location: {url} -------- N flats: {n_flats} -------- N total: {n_total}",
            outerboundaryis=list(geom.exterior.coords),
        )
        pol.style.polystyle.color = "9939FF14"
        pol.style.polystyle.outline = 1
    l = len(sample_gdf)
    fpath = os.path.join(
        PROJECT_DIR,
        "outputs",
        "data",
        f"{today}_UNLABELLED_GB_buildings_containing_flats_sample_n{l}_seed{seed}.kml",
    )
    kml.save(fpath)
    s3.Bucket(BUCKET).upload_file(
        os.path.join(os.getcwd(), fpath),
        os.path.join("outputs", "models", "block_of_flats_classifier", fpath),
    )
