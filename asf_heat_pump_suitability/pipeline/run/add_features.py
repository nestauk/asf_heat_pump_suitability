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

    # Load planned heat network zone polygons (if available for the local authority/local authorities)
    hn_zones_gdf = gpd.GeoDataFrame()
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
                f"No heat network zone geodata found for {la}. Assuming no UPRNs are in heat network zones in this Local Authority."
            )

    # Check if data is available for all LAs in the list, and if not, check if there is data for the whole set of LAs (e.g. Greater Manchester as a whole instead of individual LAs)
    if len(list_las) > 0 and hn_zones_gdf.empty:
        try:
            hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(
                local_authority=local_authorities
            )
        except ValueError:
            print(
                f"No heat network zone geodata found for {local_authorities}. Assuming no UPRNs are in heat network zones in this group of Local Authorities."
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

    buildings_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
        layer="building", grid_squares=grid_squares
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
    # ADD EPC FEATURES - EPC RATING, ATTACHMENT, TENURE, SOLAR PV and HEAT DEMAND
    epc_df = pl.read_parquet(
        config["data"]["epc"]["domestic"],
        columns=[
            "UPRN",
            "TENURE",
            "BUILT_FORM",
            "CURRENT_ENERGY_RATING",
            "SOLAR_WATER_HEATING_FLAG",
            "ENERGY_CONSUMPTION_CURRENT",
            "ENERGY_CONSUMPTION_POTENTIAL",
        ],
    )

    # Add column to indicate whether EPC data is available for the property, to distinguish between False and unknown for features derived from EPC data
    epc_df = epc_df.with_columns(pl.lit(True).alias("epc_data_available"))

    features_df = epc.extend_df_epc_features(
        df=features_df,
        epc_df=epc_df,
        columns=[
            "UPRN",
            "epc_data_available",
            "TENURE",
            "CURRENT_ENERGY_RATING",
            "SOLAR_WATER_HEATING_FLAG",
            "ENERGY_CONSUMPTION_CURRENT",
            "ENERGY_CONSUMPTION_POTENTIAL",
        ],
    )

    features_df = features_df.with_columns(
        pl.col("SOLAR_WATER_HEATING_FLAG")
        .replace({"unknown": None})
        .alias("SOLAR_WATER_HEATING_FLAG")
    )

    print(features_df["SOLAR_WATER_HEATING_FLAG"].value_counts())
    print(features_df["ENERGY_CONSUMPTION_CURRENT"].mean())
    print(features_df["ENERGY_CONSUMPTION_POTENTIAL"].mean())

    epc_with_geometries_df = epc_df.join(
        features_df.select(
            [
                pl.col("UPRN").cast(pl.Utf8),  # Cast to String to match epc_df
                "X_COORDINATE",
                "Y_COORDINATE",
            ]
        ),
        how="left",
        on="UPRN",
    )

    # Add listed building boolean flag
    from asf_heat_pump_suitability.pipeline.prepare_features import listed_buildings

    uprns_listed_buildings_df = listed_buildings.generate_df_epc_listed_buildings(
        epc_df=epc_with_geometries_df
    ).with_columns(pl.lit(True).alias("in_listed_building"))

    from asf_heat_pump_suitability.pipeline.transform.epc import retain_df_valid_uprns

    uprns_listed_buildings_df = retain_df_valid_uprns(
        uprns_listed_buildings_df, drop=True
    )

    features_df = features_df.join(
        uprns_listed_buildings_df.select(["UPRN", "in_listed_building"]),
        how="left",
        on="UPRN",
    )
    # Fill any UPRNs not in the listed buildings dataset with False (i.e. not listed) if they're in the EPC dataset, and with Null if they're not in the EPC dataset (i.e. unknown)
    features_df = features_df.with_columns(
        pl.when(pl.col("in_listed_building").is_null() & ~pl.col("epc_data_available"))
        .then(False)
        .otherwise(None)
        .alias("in_listed_building")
    )

    print(features_df["in_listed_building"].value_counts())

    del epc_df, uprns_listed_buildings_df

    # Add number of off-gas properties
    # Source: https://www.xoserve.com/help-centre/supply-points-metering/supply-point-administration-spa/
    from asf_heat_pump_suitability.pipeline.prepare_features import (
        off_gas,
    )

    off_gas_list = off_gas.process_off_gas_data()
    code_point_df = gpd.read_file(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/codepo_gb.gpkg",
        layers="codepoint",
    )
    code_point_df["POSTCODE"] = code_point_df["postcode"].str.replace(" ", "")

    nearest_postcode_df = pl.from_pandas(
        uprns_gdf.sjoin_nearest(
            code_point_df[["POSTCODE", "geometry"]],
            how="left",
            max_distance=1000,
            distance_col="distance_to_postcode_m",  # distance in metres
        ).drop(columns="index_right")[["UPRN", "POSTCODE", "distance_to_postcode_m"]]
    )

    # Label all UPRNs with on/off gas where possible
    off_gas_df = nearest_postcode_df.with_columns(
        # Label postcodes according to on/off gas
        pl.when(pl.col("POSTCODE").is_in(off_gas_list))
        .then(True)
        .otherwise(False)
        .alias("off_gas")
    ).select(["UPRN", "off_gas"])

    features_df = features_df.join(
        off_gas_df.select(["UPRN", "off_gas"]),
        how="left",
        on="UPRN",
    )

    print(features_df["off_gas"].value_counts())

    # Add near coastline boolean flag
    coast_gdf = gpd.read_file(
        "s3://asf-heat-pump-suitability/exploration/spatial_clustering_plymouth/Countries_December_2024_Boundaries_UK_BFC_6983126662299524946/CTRY_DEC_2024_UK_BFC.shp"
    )

    # Simplify coastline boundaries by 150m and buffer by 1500m to create a 'near coastline' area
    coast_gdf["simplified_geometry"] = coast_gdf["geometry"].apply(
        lambda x: x.simplify(tolerance=150).buffer(1500)
    )
    coast_gdf = coast_gdf.set_geometry("simplified_geometry")
    coast_gdf["near_coastline"] = True

    uprns_gdf.drop(columns=["index_right"], inplace=True)
    uprns_gdf = uprns_gdf.sjoin(
        coast_gdf[["near_coastline", "simplified_geometry"]],
        how="left",
        predicate="within",
    )

    features_df = features_df.join(
        pl.from_pandas(uprns_gdf[["UPRN", "near_coastline"]]),
        how="left",
        on="UPRN",
    ).with_columns(pl.col("near_coastline").fill_null(False))

    print(features_df["near_coastline"].value_counts())

    del coast_gdf

    # Add conservation area boolean flag
    from asf_heat_pump_suitability.pipeline.prepare_features import (
        off_gas,
        protected_areas,
    )

    # import boto3

    # s3_client = boto3.client('s3')

    # bucket_name = 'asf-heat-pump-suitability'
    # prefix = 'local_heat_planning/inputs/geodata/NSUL_DEC_2025/'

    # response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    # files = [
    #     f"s3://{bucket_name}/{obj['Key']}"
    #     for obj in response.get('Contents', [])
    #     if obj['Key'].endswith('.csv')
    # ]

    # uprn_to_country_df = pd.concat([pd.read_csv(file, usecols=["UPRN", "PCDS", "ctry25cd"]) for file in files], ignore_index=True)

    uprn_to_country_df = pd.DataFrame()
    import os

    folder_path = "/Users/anasofiapinto/Downloads/NSUL_DEC_2025/Data"
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    for file in files:
        uprn_to_country_df = pd.concat(
            [
                uprn_to_country_df,
                pd.read_csv(
                    os.path.join(folder_path, file),
                    usecols=["UPRN", "PCDS", "ctry25cd"],
                ),
            ],
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
    uprns_gdf = uprns_gdf.merge(
        uprn_to_country_df[["UPRN", "COUNTRY"]], on="UPRN", how="left"
    ).drop(columns=["index_right"])

    uprns_conservation_areas_df = (
        protected_areas.load_transform_df_uprn_in_protected_area(gdf=uprns_gdf)
    )

    uprns_conservation_areas_df = uprns_conservation_areas_df.with_columns(
        pl.lit(True).alias("in_conservation_area")
    )

    features_df = features_df.join(
        uprns_conservation_areas_df.select(["UPRN", "in_conservation_area"]),
        how="left",
        on="UPRN",
    ).with_columns(pl.col("in_conservation_area").fill_null(False))

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
