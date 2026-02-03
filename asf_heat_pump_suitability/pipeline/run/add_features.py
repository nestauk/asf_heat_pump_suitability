"""
Script to add features to UPRNs:
- flat / apartment property type boolean flag
- estimated max contiguous and total outdoor space (m2)

Run:
python asf_heat_pump_suitability/pipeline/run/add_features.py --uprns path/to/domestic/UPRNs

where you should replace `path/to/domestic/UPRNs` with the path to the parquet file

To run in test mode without saving outputs, add --test_mode flag.
python asf_heat_pump_suitability/pipeline/run/add_features.py --uprns path/to/domestic/UPRNs --test_mode
"""

import argparse
from asf_heat_pump_suitability import config


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    # TODO this is a placeholder and likely to change as the script develops
    parser.add_argument(
        "--uprns",
        help="Path to domestic UPRN dataset with X and Y coordinates in parquet.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--local_authorities",
        help="",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--test_mode",
        help="If set, runs in test mode without saving outputs.",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import polars as pl
    import geopandas as gpd
    import pandas as pd

    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.getters import load_tree_input
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns, outdoor_space
    from asf_heat_pump_suitability.getters.get_datasets import (
        load_gdf_inspire_land_parcels,
    )

    args = parse_arguments()
    las = args.local_authorities.lower()

    # Load UPRN data
    print(f"Loading domestic UPRNs from: {args.uprns}")
    uprns_df = pl.read_parquet(
        args.uprns, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )

    # Get geopoints of UPRNs
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # ------------------------ #
    # IMPUTE PROPERTY TYPE FLAT
    # Create boolean column called `property_type_flat` to identify flats
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    features_df = uprns_df.with_columns(
        pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat")
    )

    # ------------------------ #
    # ESTIMATE OUTDOOR SPACE
    # TODO scale beyond Plymouth. This is a temporary fix to working with multiple LAs
    print("Loading land registry data...")

    if las == "plymouth":
        land_parcels_gdf = gpd.read_file(
            "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Plymouth_Land_Registry_Cadastral_Parcels.gml"
        )
    else:
        inspire_file_names = load_gdf_inspire_land_parcels(
            path="s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/inspire_file_bounds_EW.geojson"
        )
        inspire_file_names = inspire_file_names[
            inspire_file_names["LAD23NM"].isin(config["constant"][las]["la_names"])
        ]
        inspire_file_names = inspire_file_names["inspire_file_name"].unique()
        land_parcels_gdf = gpd.GeoDataFrame()
        for file_name in inspire_file_names:
            print(f"Loading land parcels from {file_name}...")
            temp_gdf = load_gdf_inspire_land_parcels(path=f"s3://{file_name}")
            land_parcels_gdf = pd.concat(
                [land_parcels_gdf, temp_gdf], ignore_index=True
            )

    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=config["constant"][las]["grid_squares"]
    )

    # Get intersection of building footprint polygons and land polygons
    intersection_gdf = outdoor_space.generate_gdf_building_intersections(
        land_parcels_gdf=land_parcels_gdf,
        building_footprints_gdf=building_footprints_gdf,
    )

    # Get outdoor space
    outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
        building_intersections_gdf=intersection_gdf, land_parcels_gdf=land_parcels_gdf
    )
    uprns_space_df = outdoor_space.sjoin_df_uprn_to_outdoor_space(
        uprns_gdf=uprns_gdf, outdoor_space_gdf=outdoor_space_gdf
    )
    uprns_space_df = outdoor_space.deduplicate_df_outdoor_space(uprns_space_df)

    # Join outdoor space estimates to dataframe
    features_df = features_df.join(
        uprns_space_df.select(
            [
                "UPRN",
                "NATIONALCADASTRALREFERENCE",
                "max_contiguous_outdoor_space_area_m2",
                "total_outdoor_space_area_m2",
            ]
        ),
        how="left",
        on="UPRN",
    )

    # ------------------------ #
    # SAVE OUTPUTS

    if not args.test_mode:
        save_utils.save_to_s3(
            features_df,
            path=f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{os.path.basename(args.uprns).split('.')[0]}_with_features.parquet",
        )
