"""
Script to add features to UPRNs:
- flat / apartment property type boolean flag
- boolean flags to indicate whether UPRN is in a block of flats; in a heat network zone; and in a city centre
- estimated max contiguous and total outdoor space (m2)
- EPC-derived features of tenure; attachment type of property; and current energy rating
- TODO: Sofia still to add info

Run:
python asf_heat_pump_suitability/pipeline/run/add_features.py --local_authorities LOCAL_AUTHORITIES

where LOCAL_AUTHORITIES is one of:
- `plymouth` for Plymouth only
- `plymouth_similar` for Plymouth and 4 similar local authorities (Liverpool, Portsmouth, Southampton, Swansea)
- `sampling_areas` for Plymouth and 5 different local authorities for sampling buildings (Bath, Bradford, Glasgow, Manchester, Nottingham)
- `greater_manchester_las` for all Greater Manchester local authorities (Bolton, Bury, Manchester, Oldham, Rochdale, Salford, Stockport, Tameside, Trafford, Wigan)
- `cardiff` for Cardiff only
You can see the full list of local authority options in the `constant` section of the config.yaml file.

Set -- `--detail "simplified"` to use simplified spatial signature polygons to label city centres. The default is "full" which uses the fully detailed spatial signatures framework.

To save outputs to S3, add --save flag.
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
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--detail",
        help="Level of detail for spatial signatures dataset to label city centres. Takes values 'simplified' or 'full'. Defaults to 'full'.",
        required=False,
        default="full",
        type=str,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
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

    local_authorities = args.local_authorities.lower()

    list_las = (
        config["constant"][local_authorities]["la_names"]
        if isinstance(config["constant"][local_authorities]["la_names"], list)
        else [config["constant"][local_authorities]["la_names"]]
    )
    detail_level = args.detail
    uprns_path = config["output"]["dataset"]["residential_uprns"].format(
        local_authority=local_authorities
    )
    grid_squares = config["constant"][local_authorities]["grid_squares"]

    # Load UPRN data
    print(f"Loading domestic UPRNs from: {uprns_path}")
    uprns_df = pl.read_parquet(
        uprns_path,
        columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"],
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
    buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=grid_squares
    )

    # Map UPRNs to the ID of the building they're in
    uprn_building_id_dict = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=uprns_gdf, buildings_gdf=buildings_gdf, id_col="ID"
    )

    uprns_gdf["property_type_flat"] = uprns_gdf["UPRN"].isin(flat_uprns)
    # Generate features for block of flats classification model
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=buildings_gdf,
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

    del uprn_building_id_dict, building_features_df, labelled_df, clf

    # ------------------------ #
    # ADD CITY CENTRE AND HEAT NETWORK ZONE BOOLEAN FLAGS

    # Load planned heat network zone polygons
    hn_zones_gdf = gpd.GeoDataFrame()
    # Check if heat network zone geodata is available for each LA in the list, and if so, load it and concatenate it to a single geodataframe.
    for la in list_las:
        try:
            # TODO: deal with the potential for different Zone ID column names in different HN zone datasets
            hn_zones_gdf = pd.concat(
                [
                    hn_zones_gdf,
                    load_geodata.load_gdf_heat_network_zones(local_authority=la),
                ],
                ignore_index=True,
            )
        except ValueError:
            print(
                f"No heat network zone geodata found for {la}. All UPRNs will be labelled as 'outside heat network zone' in this Local Authority."
            )

        # If hn_zones_gdf is empty after attempting to load for each LA individually, try loading a combined HN zone geodataframe for the whole list of LAs
        # (this is because for some groups of LAs, e.g. Greater Manchester Combined Authority, there is only a combined HN zone geodataframe and no individual ones).
        try:
            hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(
                local_authority=local_authorities
            )
        except ValueError:
            print(
                f"No heat network zone geodata found for {local_authorities}. All UPRNs will be labelled as 'outside heat network zone' in this group of Local Authorities."
            )

    features_df = heat_network_zones.extend_df_heat_network_zone_bool(
        uprns_df=features_df, uprns_gdf=uprns_gdf, hn_zone_gdf=hn_zones_gdf
    )

    # Load spatial signature polygons and label UPRNs in city centres
    spatial_signatures_gdf = load_geodata.load_gdf_spatial_signatures_gb(
        detail_level=detail_level
    )
    features_df = city_centres.extend_df_city_centre_labels(
        uprns_df=features_df,
        uprns_gdf=uprns_gdf,
        spatial_signatures_gdf=spatial_signatures_gdf,
    )
    del hn_zones_gdf, spatial_signatures_gdf
    # ------------------------ #
    # ESTIMATE OUTDOOR SPACE
    # TODO scale beyond Plymouth. This is a temporary fix to working with multiple LAs
    print("Loading land registry data...")

    if local_authorities == "plymouth":
        land_parcels_gdf = gpd.read_file(
            "s3://asf-heat-pump-suitability/local_heat_planning/plymouth_inputs/Plymouth_Land_Registry_Cadastral_Parcels.gml"
        )
    else:
        inspire_file_names = get_datasets.load_gdf_inspire_land_parcels(
            path="s3://asf-heat-pump-suitability/outputs/2023Q4/gardens/inspire_file_bounds_EW.geojson"
        )
        inspire_file_names = inspire_file_names[
            inspire_file_names["LAD23NM"].isin(list_las)
        ]["inspire_file_name"].unique()

        land_parcels_gdf = pd.concat(
            [
                get_datasets.load_gdf_inspire_land_parcels(path=f"s3://{file}")
                for file in inspire_file_names
            ],
            ignore_index=False,
        )

    # Get intersection of building footprint polygons and land polygons
    intersection_gdf = outdoor_space.generate_gdf_building_intersections(
        land_parcels_gdf=land_parcels_gdf,
        buildings_gdf=buildings_gdf,
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

    del (
        land_parcels_gdf,
        intersection_gdf,
        outdoor_space_gdf,
        uprns_space_df,
    )

    # ------------------------ #
    # CONTEXTUAL FEATURES
    # ------------------------ #
    # ADD EPC FEATURES - EPC RATING, ATTACHMENT, TENURE, SOLAR PV info, ESTIMATED CURRENT ENERGY CONSUMPTION and POSTCODE
    epc_df = pl.read_parquet(
        config["data"]["epc"]["domestic"],
        columns=[
            "UPRN",
            "TENURE",
            "BUILT_FORM",
            "CURRENT_ENERGY_RATING",
            "SOLAR_WATER_HEATING_FLAG",
            "PHOTO_SUPPLY",
            "ENERGY_CONSUMPTION_CURRENT",
            "POSTCODE",
        ],
    )

    features_df = epc.extend_df_epc_features(
        df=features_df,
        epc_df=epc_df,
        columns=[
            "UPRN",
            "TENURE",
            "CURRENT_ENERGY_RATING",
            "SOLAR_WATER_HEATING_FLAG",
            "ENERGY_CONSUMPTION_CURRENT",
            "PHOTO_SUPPLY",
            "POSTCODE",
        ],
    )

    print(features_df["has_solar_pv"].value_counts())
    print(features_df["ENERGY_CONSUMPTION_CURRENT"].mean())

    # Add listed building boolean flag
    from asf_heat_pump_suitability.pipeline.prepare_features import (
        listed_buildings,
    )

    # Load listed buildings geodataframe for Great Britain
    listed_buildings_gdf = listed_buildings.transform_gdf_listed_buildings(nation="GB")

    features_df = listed_buildings.extend_df_listed_building_bool(
        features_df=features_df,
        uprns_gdf=uprns_gdf,
        buildings_gdf=buildings_gdf,
        listed_buildings_gdf=listed_buildings_gdf,
    )

    # listed_polygons = listed_buildings_gdf[
    #     listed_buildings_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    # ]
    # listed_points = listed_buildings_gdf[
    #     listed_buildings_gdf.geometry.type.isin(["Point", "MultiPoint"])
    # ]

    # # Join for Polygons: We want buildings that touch/overlap listed polygons
    # joined_polys = gpd.sjoin(
    #     buildings_gdf, listed_polygons, how="inner", predicate="intersects"
    # )

    # # Join for Points: We want buildings that contain the listed points
    # joined_pts = gpd.sjoin(
    #     buildings_gdf, listed_points, how="inner", predicate="contains"
    # )

    # joined = pd.concat([joined_polys, joined_pts], ignore_index=True)

    # uprns_gdf = gpd.sjoin(
    #     uprns_gdf,
    #     joined[["geometry", "in_listed_building"]],
    #     how="left",
    #     predicate="intersects",
    # ).fillna({"in_listed_building": False})

    # features_df = features_df.join(
    #     pl.from_pandas(uprns_gdf[["UPRN", "in_listed_building"]]),
    #     how="left",
    #     on="UPRN",
    # )

    print(features_df["in_listed_building"].value_counts())

    del listed_buildings_gdf

    # Add number of off-gas properties
    # Source: https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/
    from asf_heat_pump_suitability.pipeline.prepare_features import (
        off_gas,
    )

    off_gas_list = off_gas.process_off_gas_data()
    # code_point_df = gpd.read_file(
    #     config["data"]["geodata"]["gb_code_point_data"],
    #     layers="codepoint",
    # )
    # code_point_df["POSTCODE"] = code_point_df["postcode"].str.replace(" ", "")

    code_point_df = load_geodata.load_code_point_data()

    features_df = off_gas.extend_df_off_gas(
        features_df=features_df,
        uprns_gdf=uprns_gdf,
        code_point_df=code_point_df,
        off_gas_list=off_gas_list,
    )

    #  # create dictionary mapping between ID and POSTCODE when POSTCODE is not null
    # id_postcode_mapping_df = (
    #     features_df.filter(pl.col("POSTCODE").is_not_null())
    #     .select(["ID", "POSTCODE"])
    #     .rename({"POSTCODE": "MAPPED_POSTCODE"})
    # )

    # postcodes_df = (
    #     features_df.select(["UPRN", "ID"])
    #     .join(id_postcode_mapping_df, on="ID", how="left")
    #     .rename({"MAPPED_POSTCODE": "POSTCODE"})
    # )

    # missing_uprns = postcodes_df.filter(pl.col("POSTCODE").is_null()).get_column("UPRN")

    # uprns_no_postcode_gdf = uprns_gdf[uprns_gdf["UPRN"].isin(missing_uprns)]

    # nearest_postcode_df = pl.from_pandas(
    #     uprns_no_postcode_gdf.drop(columns="index_right")
    #     .sjoin_nearest(
    #         code_point_df[["POSTCODE", "geometry"]],
    #         how="left",
    #         max_distance=500,  # 500 metres to be conservative
    #         distance_col="distance_to_postcode_m",  # distance in metres
    #     )
    #     .drop(columns="index_right")[["UPRN", "POSTCODE", "distance_to_postcode_m"]]
    # )

    # uprn_postcode_map_df = pl.concat(
    #     [
    #         postcodes_df.filter(pl.col("POSTCODE").is_not_null()).select(
    #             ["UPRN", "POSTCODE"]
    #         ),
    #         nearest_postcode_df.select(["UPRN", "POSTCODE"]),
    #     ],
    #     how="vertical",
    # )
    # # Label all UPRNs with on/off gas where possible
    # off_gas_df = uprn_postcode_map_df.with_columns(
    #     # Label postcodes according to on/off gas
    #     pl.when(pl.col("POSTCODE").is_in(off_gas_list))
    #     .then(True)
    #     .otherwise(False)
    #     .alias("off_gas")
    # ).select(["UPRN", "off_gas"])

    # features_df = features_df.join(
    #     off_gas_df,
    #     how="left",
    #     on="UPRN",
    # )

    print(features_df["off_gas"].value_counts())

    # Add near coastline boolean flag
    # coast_gdf = gpd.read_file(
    #     config["data"]["geodata"]["gb_coast_boundaries"],
    # )
    # coast_gdf = gpd.GeoDataFrame(
    #     geometry=[coast_gdf.geometry.union_all()], crs=coast_gdf.crs
    # )

    coast_gdf = load_geodata.load_gb_coast_boundaries()

    from asf_heat_pump_suitability.pipeline.transform import coast

    features_df = coast.extend_df_near_coastline_bool(
        features_df=features_df,
        uprns_gdf=uprns_gdf,
        coast_gdf=coast_gdf,
        distance_threshold_m=1500,
        simplify_tolerance_m=150,
    )

    #  # Simplify coastline boundaries by 150m and buffer by 1500m to create a 'near coastline' area
    # coast_gdf["simplified_geometry"] = coast_gdf.geometry.boundary.simplify(
    #     tolerance=150
    # ).buffer(1500)
    # coast_gdf.set_geometry("simplified_geometry", inplace=True)
    # coast_gdf["within_1500m_coastline"] = True

    # uprns_gdf = uprns_gdf.drop(columns="index_right").sjoin(
    #     coast_gdf[["within_1500m_coastline", "simplified_geometry"]],
    #     how="left",
    #     predicate="within",
    # )

    # features_df = features_df.join(
    #     pl.from_pandas(uprns_gdf[["UPRN", "within_1500m_coastline"]]),
    #     how="left",
    #     on="UPRN",
    # ).with_columns(pl.col("within_1500m_coastline").fill_null(False))

    print(features_df["within_1500m_coastline"].value_counts())

    del coast_gdf

    # Add conservation area boolean flag
    from asf_heat_pump_suitability.pipeline.prepare_features import (
        off_gas,
        protected_areas,
    )

    import boto3

    def load_transform_df_uprn_to_country_mapping():
        """
        Load and transform the UPRN to country mapping data from S3.
        Returns:
            pd.DataFrame: A dataframe containing UPRN and corresponding country information.
        """
        s3_client = boto3.client("s3")

        bucket_name = "asf-heat-pump-suitability"
        prefix = "local_heat_planning/inputs/geodata/NSUL_DEC_2025/"

        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files = [
            f"s3://{bucket_name}/{obj['Key']}"
            for obj in response.get("Contents", [])
            if obj["Key"].endswith(".csv")
        ]

        uprn_to_country_df = pd.concat(
            [pd.read_csv(file, usecols=["UPRN", "PCDS", "ctry25cd"]) for file in files],
            ignore_index=True,
        )

        uprn_to_country_df["COUNTRY"] = (
            uprn_to_country_df["ctry25cd"]
            .str[0]
            .map(
                {
                    "E": "England",
                    "W": "Wales",
                    "S": "Scotland",
                }
            )
        )

        return uprn_to_country_df[["UPRN", "COUNTRY"]]

    uprn_to_country_df = load_transform_df_uprn_to_country_mapping()

    uprn_to_country_dict = dict(
        zip(uprn_to_country_df["UPRN"], uprn_to_country_df["COUNTRY"])
    )
    uprns_gdf["COUNTRY"] = uprns_gdf["UPRN"].map(uprn_to_country_dict)

    uprns_conservation_areas_df = (
        protected_areas.load_transform_df_uprn_in_protected_area(
            gdf=uprns_gdf.drop(columns="index_right")
        )
    )

    print(uprns_conservation_areas_df.columns)
    print(uprns_conservation_areas_df)

    features_df = (
        features_df.join(
            uprns_conservation_areas_df.select(["UPRN", "in_protected_area"]),
            how="left",
            on="UPRN",
        )
        .with_columns(pl.col("in_protected_area").fill_null(False))
        .rename({"in_protected_area": "in_conservation_area"})
    )

    print(features_df["in_conservation_area"].value_counts())

    del uprns_conservation_areas_df

    # ------------------------ #
    # SAVE OUTPUTS

    if args.save:
        save_utils.save_to_s3(
            features_df,
            path=config["output"]["dataset"]["residential_uprns_with_features"].format(
                local_authority=local_authorities
            ),
        )
