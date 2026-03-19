"""
Script to add features to UPRNs:
- flat / apartment property type boolean flag
- boolean flag to indicate whether UPRN is in a block of flats
- estimated max contiguous and total outdoor space (m2)

Run:
python asf_heat_pump_suitability/pipeline/run/add_features.py --uprns path/to/domestic/UPRNs.parquet

To save outputs to S3, add --save flag:
python asf_heat_pump_suitability/pipeline/run/add_features.py --uprns path/to/domestic/UPRNs.parquet --save
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
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import polars as pl
    import geopandas as gpd
    import pandas as pd

    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.getters import (
        load_tree_input,
        base_getters,
        load_geodata,
    )
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.model.block_of_flats import (
        feature_engineering,
        train_model,
    )
    from asf_heat_pump_suitability.pipeline.transform import (
        uprns,
        outdoor_space,
        epc,
        heat_network_zones,
        city_centres,
    )
    from asf_heat_pump_suitability.getters import get_datasets

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
    # PREDICT BLOCK OF FLATS CLASSIFICATION
    # Load building footprint data
    # TODO scale beyond sampling areas
    building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares="SX"
    )

    # Map UPRNs to the ID of the building they're in
    uprn_building_id_dict = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=building_footprints_gdf, id_col="ID"
    )

    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)
    # Generate features for block of flats classification model
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=building_footprints_gdf,
        uprns_gdf=uprns_gdf,
        id_col="ID",
    )

    # TODO make this robust by automatically loading same labelled data used to train model
    labelled_df = pl.read_parquet(
        "s3://asf-heat-pump-suitability/local_heat_planning/inputs/processed/manually_labelled_block_of_flats.parquet"
    )
    # Load trained block of flats classifier model
    clf = base_getters.load_pickle(
        config["output"]["save_as"]["model"]["block_of_flats_model"]
    )
    features_df = train_model.extend_df_in_block_of_flats_label(
        uprns_df=features_df,
        mapping=uprn_building_id_dict,
        predictions_df=train_model.predict_class_block_of_flats(
            model=clf,
            features_df=building_features_df,
            labelled_df=labelled_df,
            id_col="ID",
        ),
        id_col="ID",
    )

    # ------------------------ #
    # ADD CITY CENTRE AND HEAT NETWORK ZONE BOOLEAN FLAGS
    # Load Plymouth existing heat network zone polygons and label UPRNs in HN zones
    hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(local_authority=las)
    features_df = heat_network_zones.extend_df_heat_network_zone_bool(
        uprns_df=features_df, uprns_gdf=uprns_gdf, hn_zone_gdf=hn_zones_gdf
    )

    # Load spatial signature polygons and label UPRNs in city centres
    spatial_signatures_gdf = load_geodata.load_gdf_spatial_signatures_gb(
        detail_level="full"
    )
    features_df = city_centres.extend_df_city_centre_labels(
        uprns_df=features_df,
        uprns_gdf=uprns_gdf,
        spatial_signatures_gdf=spatial_signatures_gdf,
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
        inspire_file_names = get_datasets.load_gdf_inspire_land_parcels(
            path="s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/inspire_file_bounds_EW.geojson"
        )
        inspire_file_names = inspire_file_names[
            inspire_file_names["LAD23NM"].isin(config["constant"][las]["la_names"])
        ]["inspire_file_name"].unique()

        land_parcels_gdf = pd.concat(
            [
                get_datasets.load_gdf_inspire_land_parcels(path=f"s3://{file}")
                for file in inspire_file_names
            ],
            ignore_index=False,
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
    # PRIORITISATION FEATURES
    # ------------------------ #
    # ADD EPC FEATURES - EPC RATING, ATTACHMENT, TENURE
    epc_df = pl.read_parquet(config["data"]["epc"]["domestic"])
    uprns_df = epc.extend_df_epc_features(
        uprns_df=uprns_df,
        epc_df=epc_df,
        columns=["UPRN", "ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"],
    )

    # ------------------------ #
    # SAVE OUTPUTS

    if args.save:
        save_utils.save_to_s3(
            features_df,
            path=f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{os.path.basename(args.uprns).split('.')[0]}_with_features.parquet",
        )
