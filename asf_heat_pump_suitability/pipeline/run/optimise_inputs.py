"""
Script to optimise input datasets for speed and memory gains during pipeline runs. Converts heavy file types into
more efficient parquet files and, where possible, partitions geospatial data by grid square.

To run (all grid squares):
    python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py

To run for specific grid squares only (e.g. to re-process or test):
    python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py --grid_squares SX SY

To run for specific datasets only:
    python asf_heat_pump_suitability/pipeline/run/optimise_inputs.py --datasets EPC POI
"""

import argparse
import warnings

from typing import List
from collections import defaultdict

import polars as pl
import polars.selectors as cs

import s3fs

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_geodata
from asf_heat_pump_suitability.utils import save_utils, geo_utils


def build_df_grid_square_lookup() -> pl.DataFrame:
    """
    Build a Polars lookup table mapping 100km BNG tiles to grid square letter codes (e.g. SX, NT).

    Returns:
        pl.DataFrame: lookup with easting_100km and northing_100km coordinates to their corresponding grid square values
    """
    grid_gdf = load_geodata.load_gdf_bng_grid_squares()
    bounds = grid_gdf.geometry.bounds
    # Convert easting and northing to single digit integers (100km values)
    return pl.DataFrame(
        {
            "easting_100km": (bounds["minx"] // 100000).astype(int).tolist(),
            "northing_100km": (bounds["miny"] // 100000).astype(int).tolist(),
            "grid_square": grid_gdf["bng_ref"].tolist(),
        }
    ).with_columns(cs.numeric().cast(pl.Int32))


def assign_df_grid_squares(
    df: pl.DataFrame,
    x_col: str = "X",
    y_col: str = "Y",
    gs_lookup: pl.DataFrame = None,
) -> pl.DataFrame:
    """
    Join a grid square column to a UPRN DataFrame using X and Y coordinate information.

    Args:
        df (pl.DataFrame): UPRN data with X_COORDINATE and Y_COORDINATE columns in BNG (EPSG:27700).
        gs_lookup (pl.DataFrame): grid square lookup with easting_100km and northing_100km corresponding to grid square values.
        Optional. If None, defaults to generating new grid square lookup.

    Returns:
        pl.DataFrame: UPRNs with their corresponding grid square
    """
    if gs_lookup is None:
        gs_lookup = build_df_grid_square_lookup()
    df = (
        df.with_columns(
            [
                # Convert X and Y coordinates to single digit easting and northing values
                (pl.col(x_col) // 100000).cast(pl.Int32).alias("easting_100km"),
                (pl.col(y_col) // 100000).cast(pl.Int32).alias("northing_100km"),
            ]
        )
        .join(gs_lookup, on=["easting_100km", "northing_100km"], how="left")
        .drop(["easting_100km", "northing_100km"])
    )

    null_count = df["grid_square"].null_count()
    if null_count > 0:
        warnings.warn(
            f"{null_count} UPRNs could not be assigned to a grid square and will be skipped."
        )
        df = df.drop_nulls(subset=["grid_square"])
    return df


def partition_geofile_to_grid_squares(
    df: pl.DataFrame | List[pl.DataFrame],
    grid_squares: list,
    fpath: str,
    multi_file: bool = False,
) -> None:
    """
    Partition a single dataframe containing geospatial information into GB 100km grid squares and save to S3.

    Args:
        df (pl.DataFrame): dataframe or list of dataframes to partition containing `grid_square` column
        grid_squares (list): all grid squares contained in `df`
        fpath (str): generic path to save partitioned files to. String must contain `{grid_square}` for formatting corresponding
        grid square for each file.
        multi_file (bool): set to True if multiple source files are being partitioned. This handles grid squares spanning
        across several files. Default False.
    """
    if multi_file:
        if not isinstance(df, list):
            raise TypeError(
                "No list of dataframes detected despite setting `multi-file` to True. Set to False or enter a list of dataframes for processing."
            )

        existing_gs = {}
        existing_suffixes = defaultdict(int)

        for chunk_df in df:
            grid_squares = chunk_df["grid_square"].unique()
            for grid_square in grid_squares:
                partition = chunk_df.filter(pl.col("grid_square") == grid_square).drop(
                    "grid_square"
                )
                if grid_square in existing_gs:
                    existing_suffixes[grid_square] += 1
                    fsuffix = grid_square + "_" + existing_suffixes["grid_square"]
                else:
                    fsuffix = grid_square
                print(
                    f"Saving {len(partition):,} rows for grid square {grid_square} to {fpath.format(grid_square=fsuffix)}..."
                )
                save_utils.save_to_s3(partition, fpath.format(grid_square=fsuffix))
            del chunk_df

        # This is required if grid squares span across multiple files.
        # It ensures there is one file per grid square at the end.
        _flatten_grid_square_files(
            suffixes=existing_suffixes, fpath=fpath, clean_up=True
        )

    else:
        # Extract dataframe from list if there is just a single one
        if isinstance(df, list) and len(df) == 1:
            df = df[0]
        for grid_square in grid_squares:
            partition = df.filter(pl.col("grid_square") == grid_square).drop(
                "grid_square"
            )
            print(
                f"Saving {len(partition):,} rows for grid square {grid_square} to {fpath.format(grid_square=grid_square)}..."
            )
            save_utils.save_to_s3(partition, fpath.format(grid_square=grid_square))


def _flatten_grid_square_files(suffixes: dict, fpath: str, clean_up: bool = True):
    """
    Flatten multiple data files for the same grid square with different suffixes (e.g. SX_1, SX_2,..., SX_n) into a
    single grid square file (e.g. SX) and saves it to S3.

    Args:
        suffixes (dict): where keys are grid square references (e.g. SX, NT) and values are integers representing the maximum
        suffix (e.g. this would be '3' if there are 3 additional files for a given grid square).
        fpath (str): generic S3 path to save partitioned files to. String must contain `{grid_square}` for formatting corresponding
        grid square for each file.
        clean_up (bool): remove redundant files from S3 after flattening. Default True.
    """
    print("Aggregating multiple files created for the same grid squares...")
    if clean_up:
        fs = s3fs.S3FileSystem()
    for grid_square in suffixes.keys():
        # Load main grid square file to append the others to
        dfs = [pl.read_parquet(fpath.format(grid_square=grid_square))]

        # Generate all suffixes for the grid square
        gs_suffixes = list(range(1, suffixes[grid_square] + 1, 1))

        # Load all files for that grid square and save a final concatenated single file for the grid square
        fsuffixes = [grid_square + "_" + suffix for suffix in gs_suffixes]
        suffix_paths = [fpath.format(grid_square=fsuffix) for fsuffix in fsuffixes]
        dfs.extend([pl.read_parquet(fpath) for fpath in suffix_paths])
        print(
            f"Concatenating {len(dfs)} files for grid square {grid_square} and saving to {fpath.format(grid_square=grid_square)}..."
        )
        save_utils.save_to_s3(
            df=pl.concat(dfs), path=fpath.format(grid_square=grid_square)
        )
        del dfs

        if clean_up:
            # Delete the files with suffixes to clean up S3 folder
            print(f"Removing redundant files after concatenation: {suffix_paths}")
            fs.rm(suffix_paths)


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
        help="Datasets to process of: UPRN; UPRN_lookup; POI; EPC_domestic; EPC_commercial. Defaults to all datasets with optimisation options.",
        type=str,
        nargs="+",
        default=None,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    import boto3
    from asf_heat_pump_suitability.getters import base_getters
    from asf_heat_pump_suitability.utils import s3_utils

    args = parse_arguments()
    datasets = args.datasets
    grid_squares = (
        args.grid_squares or load_geodata.load_gdf_bng_grid_squares()["bng_ref"]
    )

    gs_lookup = build_df_grid_square_lookup()

    if not datasets or "UPRN_lookup" in datasets:
        s3_client = boto3.client("s3")
        path = config["data"]["geodata"]["gb_uprn_country_mapping"]
        bucket_name = path.split("s3://")[1].split("/")[0]
        prefix = path.split(f"s3://{bucket_name}/")[1]
        uprn_lookup_files = s3_utils.fetch_list_file_paths_from_s3_folder(
            s3_client=s3_client,
            s3_bucket=bucket_name,
            path_folder=prefix,
            file_type=".csv",
        )

        uprn_lookup_dfs = []
        for file in uprn_lookup_files:
            fpath = f"s3://asf-local-heat-planning-tool/{file}"
            print(f"Loading UPRN lookup from: {fpath}")
            uprn_lookup_df = pl.read_csv(fpath)
            uprn_lookup_dfs.append(
                assign_df_grid_squares(
                    df=uprn_lookup_df,
                    x_col="GRIDGB1E",
                    y_col="GRIDGB1N",
                    gs_lookup=gs_lookup,
                )
            )
        s3_fname = config["data"]["geodata"]["gb_uprn_lookup_partitioned"]
        partition_geofile_to_grid_squares(
            df=uprn_lookup_dfs,
            grid_squares=grid_squares,
            fpath=s3_fname,
            multi_file=True,
        )

    if not datasets or "UPRN" in datasets or "EPC_domestic" in datasets:
        uprns_df = load_geodata.load_df_osopen_uprn(parquet=False)

        # Generate parquet file
        save_utils.save_to_s3(
            uprns_df, config["data"]["geodata"]["uk_osopen_uprn_parquet"]
        )
        uprns_df = assign_df_grid_squares(
            uprns_df,
            x_col="X_COORDINATE",
            y_col="Y_COORDINATE",
            gs_lookup=gs_lookup,
        )

        if not datasets or "UPRN" in datasets:
            s3_fname = config["data"]["geodata"]["uk_osopen_uprn_partitioned"]
            partition_geofile_to_grid_squares(
                df=uprns_df, grid_squares=grid_squares, fpath=s3_fname
            )

        if not datasets or "EPC_domestic" in datasets:
            # Some UPRNs don't have coordinates so they will be dropped here.
            # This is fine because we can't use them in the pipeline because we don't know where they are located.
            # Note this would change if we use address to match UPRNs to buildings.
            epc_df = (
                base_getters.load_df_from_s3(config["data"]["epc"]["domestic"])
                .with_columns(
                    pl.col("UPRN")
                    .cast(pl.Float64, strict=False)
                    .cast(pl.Int64)
                    .alias("UPRN")
                )
                .join(
                    uprns_df.select(
                        ["UPRN", "X_COORDINATE", "Y_COORDINATE", "grid_square"]
                    ),
                    how="left",
                    on="UPRN",
                )
            )
            s3_fname = config["data"]["epc"]["domestic_partitioned"]
            partition_geofile_to_grid_squares(
                df=epc_df, grid_squares=grid_squares, fpath=s3_fname
            )
            del epc_df
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
            df=poi_df, grid_squares=grid_squares, fpath=s3_fname
        )
        del poi_df

    if not datasets or "EPC_commercial" in datasets:
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
