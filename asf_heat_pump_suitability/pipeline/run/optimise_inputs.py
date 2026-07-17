"""
Converts raw OS Open UPRN data from ZIP+CSV to Hive-partitioned Parquet by 100km BNG grid square.

This is a one-time preprocessing step. The partitioned output enables fast spatial reads by grid square,
so downstream pipeline runs load only the UPRNs relevant to the target local authority rather than
the full ~36M-row GB dataset.

To run (all grid squares):
    python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py

To run for specific grid squares only (e.g. to re-process or test):
    python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py --grid_squares SX SY
"""

import argparse
import logging

import geopandas as gpd
import polars as pl
import polars.selectors as cs
from osbng import grids

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_geodata
from asf_heat_pump_suitability.utils import save_utils, geo_utils


def build_df_grid_square_lookup() -> pl.DataFrame:
    """
    Build a Polars lookup table mapping 100km BNG tile indices to grid square reference codes.

    The BNG letter codes (e.g. SX, NT) don't follow a simple arithmetic formula, so this derives
    the mapping from the osbng grid square bounds and joins it against integer tile indices computed
    from raw easting/northing coordinates.

    Returns:
        pl.DataFrame: lookup with columns easting_100km (Int32), northing_100km (Int32), grid_square (Utf8)
    """
    grid_gdf = load_geodata.load_gdf_bng_grid_squares()
    bounds = grid_gdf.geometry.bounds
    return pl.DataFrame(
        {
            "easting_100km": (bounds["minx"] / 100000).astype(int).tolist(),
            "northing_100km": (bounds["miny"] / 100000).astype(int).tolist(),
            "grid_square": grid_gdf["bng_ref"].tolist(),
        }
    ).with_columns(cs.numeric().cast(pl.Int32))


def assign_df_grid_squares(
    df: pl.DataFrame, x_col: str = "X", y_col: str = "Y"
) -> pl.DataFrame:
    """
    Add a grid_square column to a UPRN DataFrame by joining against the 100km BNG tile lookup.

    Args:
        df (pl.DataFrame): UPRN data with X_COORDINATE and Y_COORDINATE columns in BNG (EPSG:27700).

    Returns:
        pl.DataFrame: input DataFrame with an additional grid_square column (Utf8). Rows outside
        the BNG extent will have a null grid_square.
    """
    lookup = build_df_grid_square_lookup()
    return (
        df.with_columns(
            [
                (pl.col(x_col) // 100000).cast(pl.Int32).alias("easting_100km"),
                (pl.col(y_col) // 100000).cast(pl.Int32).alias("northing_100km"),
            ]
        )
        .join(lookup, on=["easting_100km", "northing_100km"], how="left")
        .drop(["easting_100km", "northing_100km"])
    )


def partition_geofile_to_grid_squares(df: pl.DataFrame, grid_squares: list, fname: str):
    for grid_square in grid_squares:
        partition = df.filter(pl.col("grid_square") == grid_square).drop("grid_square")
        print(
            f"Saving {len(partition):,} rows for grid square {grid_square} to {fname.format(grid_square=grid_square)}..."
        )
        save_utils.save_to_s3(partition, fname.format(grid_square=grid_square))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_squares",
        help="100km BNG grid squares to process (e.g. SX SY). Defaults to all grid squares present in the data.",
        type=str,
        nargs="+",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--datasets",
        help="Datasets to process of: UPRN, POI, EPC. Defaults to all datasets with optimisation options.",
        type=str,
        nargs="+",
        default=None,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability.getters import base_getters

    args = parse_arguments()
    datasets = args.datasets
    grid_squares = (
        args.grid_squares or load_geodata.load_gdf_bng_grid_squares()["bng_ref"]
    )

    if not datasets or "UPRN" in datasets:
        uprns_df = load_geodata.load_df_osopen_uprn(parquet=False)

        # Generate parquet file
        save_utils.save_to_s3(
            uprns_df, config["data"]["geodata"]["uk_osopen_uprn_parquet"]
        )
        uprns_df = assign_df_grid_squares(
            uprns_df, x_col="X_COORDINATE", y_col="Y_COORDINATE"
        )

        null_count = uprns_df["grid_square"].null_count()
        if null_count > 0:
            logging.warning(
                f"{null_count} UPRNs could not be assigned to a grid square and will be skipped."
            )
            uprns_df = uprns_df.drop_nulls(subset=["grid_square"])

        s3_fname = config["data"]["geodata"]["uk_osopen_uprn_partitioned"]
        partition_geofile_to_grid_squares(
            df=uprns_df, grid_squares=grid_squares, fname=s3_fname
        )
        del uprns_df

    if not datasets or "POI" in datasets:
        # POI data
        poi_gdf = load_geodata.load_gdf_poi(parquet=False).to_crs(
            config["constant"]["target_crs"]
        )
        poi_df = geo_utils.convert_gdf_to_df(poi_gdf)
        poi_df = assign_df_grid_squares(poi_df)
        save_utils.save_to_s3(
            poi_df, config["data"]["geodata"]["UK_poi_locations_parquet"]
        )

        s3_fname = config["data"]["geodata"]["UK_poi_locations_partitioned"]
        partition_geofile_to_grid_squares(
            df=poi_df, grid_squares=grid_squares, fname=s3_fname
        )
        del poi_df

    if not datasets or "EPC" in datasets:
        # England and Wales register
        commercial_epc_df = base_getters.load_df_from_s3(
            config["data"]["epc"]["commercial"]["EW"],
            infer_schema_length=10000,
        )
        save_utils.save_to_s3(
            commercial_epc_df, config["data"]["epc"]["commercial"]["EW_parquet"]
        )

        # Scotland register
        commercial_epc_df = base_getters.load_df_from_s3(
            config["data"]["epc"]["commercial"]["S"]
        )
        save_utils.save_to_s3(
            commercial_epc_df, config["data"]["epc"]["commercial"]["S_parquet"]
        )
