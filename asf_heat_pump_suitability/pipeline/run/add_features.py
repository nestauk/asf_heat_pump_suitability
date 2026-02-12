"""
Script to add features to UPRNs:
- flat / apartment property type boolean flag
-
- estimated max contiguous and total outdoor space (m2)

Run:
python asf_heat_pump_suitability/pipeline/run/add_features.py --uprns path/to/domestic/UPRNs
"""

import argparse


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

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import polars as pl
    import geopandas as gpd

    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.getters import load_tree_input, base_getters
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.model.block_of_flats import (
        feature_engineering,
        train_model,
    )
    from asf_heat_pump_suitability.pipeline.transform import uprns, outdoor_space

    args = parse_arguments()

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
    # PREDICT BLOCK OF FLATS CLASSIFICATION
    # Load building footprint data
    # TODO scale beyond sampling areas
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares="SX"
    )

    uprn_building_id_dict = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=building_footprints_gdf, id_col="ID"
    )

    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    # TODO make this robust by loading same labelled data used to train model
    labelled_df = pl.read_parquet(
        "s3://asf-heat-pump-suitability/local_heat_planning/inputs/processed/manually_labelled_block_of_flats.parquet"
    )
    rfc = base_getters.load_pickle(
        config["output"]["save_as"]["model"]["block_of_flats_model"]
    )
    features_df = train_model.extend_df_in_block_of_flats_label(
        uprns_df=features_df,
        mapping=uprn_building_id_dict,
        predictions_df=train_model.predict_class_block_of_flats(
            model=rfc,
            features_df=building_features_df,
            labelled_df=labelled_df,
            id_col="ID",
        ),
        id_col="ID",
    )

    # ------------------------ #
    # ESTIMATE OUTDOOR SPACE
    # TODO scale beyond Plymouth
    print("Loading land registry data...")
    land_parcels_gdf = gpd.read_file(
        "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Plymouth_Land_Registry_Cadastral_Parcels.gml"
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
    save_utils.save_to_s3(
        features_df,
        path=f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{os.path.basename(args.uprns).split('.')[0]}_with_features.parquet",
    )
