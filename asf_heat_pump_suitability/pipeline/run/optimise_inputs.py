"""
Converts raw OS Open UPRN data from ZIP+CSV to Hive-partitioned Parquet by 100km BNG grid square.

This is a one-time preprocessing step. The partitioned output enables fast spatial reads by grid square,
so downstream pipeline runs load only the UPRNs relevant to the target local authority rather than
the full ~36M-row GB dataset.

To run (all grid squares):
    python asf_heat_pump_suitability/pipeline/run/process_inputs.py

To run for specific grid squares only (e.g. to re-process or test):
    python asf_heat_pump_suitability/pipeline/run/process_inputs.py --grid_squares SX SY
"""

import argparse
import logging

import geopandas as gpd
import polars as pl
from osbng import grids

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_geodata
from asf_heat_pump_suitability.utils import save_utils


def build_grid_square_lookup() -> pl.DataFrame:
    """
    Build a Polars lookup table mapping 100km BNG tile indices to grid square reference codes.

    The BNG letter codes (e.g. SX, NT) don't follow a simple arithmetic formula, so this derives
    the mapping from the osbng grid square bounds and joins it against integer tile indices computed
    from raw easting/northing coordinates.

    Returns:
        pl.DataFrame: lookup with columns easting_100km (Int32), northing_100km (Int32), grid_square (Utf8)
    """
    grid_gdf = gpd.GeoDataFrame.from_features(list(grids.bng_grid_100km), crs=27700)
    bounds = grid_gdf.geometry.bounds
    return pl.DataFrame(
        {
            "easting_100km": (bounds["minx"] / 100000).astype(int).tolist(),
            "northing_100km": (bounds["miny"] / 100000).astype(int).tolist(),
            "grid_square": grid_gdf["bng_ref"].tolist(),
        }
    )


def assign_grid_squares(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a grid_square column to a UPRN DataFrame by joining against the 100km BNG tile lookup.

    Args:
        df (pl.DataFrame): UPRN data with X_COORDINATE and Y_COORDINATE columns in BNG (EPSG:27700).

    Returns:
        pl.DataFrame: input DataFrame with an additional grid_square column (Utf8). Rows outside
        the BNG extent will have a null grid_square.
    """
    lookup = build_grid_square_lookup()
    return (
        df.with_columns(
            [
                (pl.col("X_COORDINATE") // 100000)
                .cast(pl.Int32)
                .alias("easting_100km"),
                (pl.col("Y_COORDINATE") // 100000)
                .cast(pl.Int32)
                .alias("northing_100km"),
            ]
        )
        .join(lookup, on=["easting_100km", "northing_100km"], how="left")
        .drop(["easting_100km", "northing_100km"])
    )


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    uprns_df = load_geodata.load_df_osopen_uprn(parquet=False)

    # Generate parquet file
    save_utils.save_to_s3(uprns_df, config["data"]["geodata"]["uk_osopen_uprn_parquet"])
    uprns_df = assign_grid_squares(uprns_df)

    null_count = uprns_df["grid_square"].null_count()
    if null_count > 0:
        logging.warning(
            f"{null_count} UPRNs could not be assigned to a grid square and will be skipped."
        )
        uprns_df = uprns_df.drop_nulls(subset=["grid_square"])

    grid_squares = args.grid_squares or sorted(
        uprns_df["grid_square"].unique().to_list()
    )

    for grid_square in grid_squares:
        partition = uprns_df.filter(pl.col("grid_square") == grid_square).drop(
            "grid_square"
        )
        path = config["data"]["geodata"]["uk_osopen_uprn_partitioned"].format(
            grid_square=grid_square
        )
        print(
            f"Saving {len(partition):,} UPRNs for grid square {grid_square} to {path}..."
        )
        save_utils.save_to_s3(partition, path)
