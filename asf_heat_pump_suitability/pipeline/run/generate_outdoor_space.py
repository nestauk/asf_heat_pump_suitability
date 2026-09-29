"""
Script to generate outdoor space estimations for each domestic UPRN in GB (or a subset of grid squares) using land extent and building footprint data.

It takes the following arguments:
    --save: If set, saves outputs to S3. If not set, outputs are not saved.
    --release_date: Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.
    --grid_squares: Comma-separated list of grid squares to process (default is None for all GB grid squares from config).

You can run this script from the command line as follows:
    python generate_outdoor_space.py --save --release_date YYYYMMDD --grid_squares grid_square1,grid_square2,...

For GB-wide processing, you can omit the --grid_squares argument or set it to None, as below:
    python generate_outdoor_space.py --save --release_date YYYYMMDD --grid_squares None
    python generate_outdoor_space.py --save --release_date YYYYMMDD
"""

import argparse
import polars as pl

# Distance (m) to buffer around a grid square's land parcels when selecting
# which buildings to load, so parcels near a grid square boundary are still
# intersected against buildings that physically sit in the neighbouring square.
BUILDING_BUFFER_M = 500


def _get_list_neighbouring_grid_squares(
    grid_square_lookup: pl.DataFrame, square: str, valid_squares: list[str]
) -> list[str]:
    """
    Return `square` plus its immediate 100km neighbours that have GB land data.

    Args:
        grid_square_lookup (pl.DataFrame): DataFrame with columns "grid_square", "easting_100km", "northing_100km" for all GB grid squares with land data
        square (str): 2-letter grid square code to find neighbours for
        valid_squares (list[str]): List of valid grid squares to consider

    Returns:
        list[str]: List of grid squares including `square` and its immediate 100km neighbours
    """
    row = grid_square_lookup.filter(pl.col("grid_square") == square)
    easting, northing = row["easting_100km"][0], row["northing_100km"][0]
    neighbour_coords = [
        {"easting_100km": easting + de, "northing_100km": northing + dn}
        for de in (-1, 0, 1)
        for dn in (-1, 0, 1)
    ]
    neighbours = grid_square_lookup.filter(
        pl.struct(["easting_100km", "northing_100km"]).is_in(neighbour_coords)
    )["grid_square"].to_list()
    return [s for s in neighbours if s in valid_squares]


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    parser.add_argument(
        "--grid_squares",
        help="Comma-separated list of grid squares to process (default is None for all GB grid squares from config).",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    import polars as pl
    import geopandas as gpd
    import pandas as pd
    import shapely

    from asf_heat_pump_suitability import config

    from asf_heat_pump_suitability.getters import (
        load_geodata,
        base_getters,
    )
    from asf_heat_pump_suitability.pipeline.transform import (
        uprns,
        outdoor_space,
    )
    from asf_heat_pump_suitability.pipeline.run import optimise_inputs
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    grid_squares = (
        args.grid_squares.split(",")
        if args.grid_squares
        else config["constant"]["land_grid_squares"]
    )

    interim_dir = (
        f"s3://asf-local-heat-planning-tool/outputs/data/GB/{args.release_date}/"
        "interim/GB_domestic_uprns_outdoor_space/"
    )
    final_path = (
        f"s3://asf-local-heat-planning-tool/outputs/data/GB/{args.release_date}/"
        "GB_domestic_uprns_outdoor_space.parquet"
    )

    uprns_path = f"s3://asf-local-heat-planning-tool/outputs/data/GB/{args.release_date}/GB_domestic_uprns.parquet"

    uprns_df = pl.read_parquet(
        uprns_path,
        columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"],
    )
    uprns_df = optimise_inputs.assign_df_grid_squares(
        uprns_df, x_col="X_COORDINATE", y_col="Y_COORDINATE"
    )

    print("Loading land registry file index...")
    inspire_file_gdf = gpd.read_parquet(
        config["data"]["processed"]["inspire_file_names"]
    )

    # grid square lookup to find neighbouring squares
    grid_square_lookup = optimise_inputs.build_df_grid_square_lookup()

    for square in grid_squares:
        square_uprns_df = uprns_df.filter(pl.col("grid_square") == square)

        # Get list of neighbouring grid squares (in addition to this square) to load building footprints from,
        # so that buildings in neighbouring squares that intersect with this square's land parcels are included
        neighbouring_grid_squares_list = _get_list_neighbouring_grid_squares(
            grid_square_lookup, square=square, valid_squares=grid_squares
        )

        if square_uprns_df.is_empty():
            print(f"No UPRNs found in {square}, skipping")
            continue

        print(f"Processing grid square {square} ({square_uprns_df.height} UPRNs)...")

        uprns_gdf = uprns.generate_gdf_uprn_coords(df=square_uprns_df)

        # Find INSPIRE files which intersect with this square's UPRNs
        inspire_file_names = inspire_file_gdf.sjoin(
            uprns_gdf, how="inner", predicate="intersects"
        )["inspire_file_name"].unique()

        if len(inspire_file_names) == 0:
            print(f"No land registry files intersect {square}, skipping")
            continue

        land_parcels_gdf = pd.concat(
            [
                outdoor_space.load_transform_gdf_land_parcels(f"s3://{file}")
                for file in inspire_file_names
            ],
            ignore_index=False,
        )
        land_parcels_gdf["geometry"] = land_parcels_gdf.normalize()
        land_parcels_gdf = land_parcels_gdf.drop_duplicates(subset=["geometry"])

        buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
            layer="building", grid_squares=neighbouring_grid_squares_list
        )

        if buildings_gdf.empty:
            print(f"No building footprints found for {square}, skipping")
            continue

        # Restrict buildings to a buffer around this `square`'s own land parcels
        buffered_bbox = shapely.box(*land_parcels_gdf.total_bounds).buffer(
            BUILDING_BUFFER_M
        )
        buildings_gdf = buildings_gdf[buildings_gdf.intersects(buffered_bbox)]

        intersection_gdf = outdoor_space.generate_gdf_building_intersections(
            land_parcels_gdf=land_parcels_gdf,
            buildings_gdf=buildings_gdf,
        )
        outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
            building_intersections_gdf=intersection_gdf,
            land_parcels_gdf=land_parcels_gdf,
        )
        square_space_df = outdoor_space.sjoin_df_uprn_to_outdoor_space(
            uprns_gdf=uprns_gdf, outdoor_space_gdf=outdoor_space_gdf
        )
        square_space_df = outdoor_space.deduplicate_df_outdoor_space(square_space_df)

        if args.save:
            save_utils.save_to_s3(square_space_df, f"{interim_dir}{square}.parquet")

        del (
            uprns_gdf,
            land_parcels_gdf,
            buildings_gdf,
            intersection_gdf,
            outdoor_space_gdf,
            square_space_df,
        )

    if args.save:
        print("Merging interim outputs into final file...")
        interim_files = base_getters.list_obj_s3_location(interim_dir)
        combined_df = pl.concat([pl.read_parquet(f"s3://{f}") for f in interim_files])
        save_utils.save_to_s3(combined_df, final_path)
