"""
Script to add features to UPRNs:
- flat / apartment property type boolean flag
- boolean flags to indicate whether UPRN is in a block of flats; in a listed building; off-gas; near the coast; and in a protected area
- estimated max contiguous and total outdoor space (m2)
- EPC-derived features of tenure; attachment type of property; and current energy rating, solar PV info, estimated current energy consumption.

Run:
python asf_heat_pump_suitability/pipeline/run/add_features.py --local_authorities LOCAL_AUTHORITIES --model_release YYYYMMDD

Set --model_release to the YYYYMMDD date of the block of flats model training run. This is used to load the trained
block of flats classifier and the manually labelled data it was trained on from the corresponding dated directory.

To save outputs to S3, add --save flag.

Set --release_date to specify the YYYYMMDD dated release directory to read inputs from and
save outputs to. Defaults to running the pipeline using today's date. Multi-day runs
should pass the same --release_date to every stage.
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
        help="Local authority or authorities (case insensitive) e.g. -- 'plymouth' to run for Plymouth or --'glasgow city' 'south lanarkshire' to run for both Glasgow City and South Lanarkshire.",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--model_release",
        help="Date of model training in YYYYMMDD format.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    import polars as pl
    import geopandas as gpd
    import pandas as pd

    from asf_heat_pump_suitability import config

    from asf_heat_pump_suitability.utils import manifest_utils, save_utils
    from asf_heat_pump_suitability.getters import (
        base_getters,
        load_data,
        load_geodata,
        load_boundaries,
    )
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.model.block_of_flats import (
        feature_engineering,
        train_model,
    )
    from asf_heat_pump_suitability.pipeline.transform import (
        barriers,
        uprns,
        outdoor_space,
        epc,
        coast,
        lookups,
        local_authority,
        listed_buildings,
        off_gas,
        protected_areas,
    )

    args = parse_arguments()

    local_authorities = [la.lower() for la in args.local_authorities]

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    release_date = save_utils.get_str_release_date(args.release_date)
    domestic_uprns_path = save_utils.get_str_output_path(
        "domestic_uprns",
        release_date=release_date,
        check_exists=True,
        local_authority=local_authority_dict["url_slug"],
    )

    # Load UPRN data
    print(f"Loading domestic UPRNs from: {domestic_uprns_path}")
    domestic_uprns_df = pl.read_parquet(
        domestic_uprns_path,
        columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"],
    )

    # ADD POSTCODE AND COUNTRY CODE FROM UPRN LOOKUP
    uprn_lookup_df = load_data.load_df_uprn_lookup(
        grid_squares=local_authority_dict["grid_squares"],
        columns=["UPRN", "PCDS", "ctry25cd"],
    )
    uprn_lookup_df = lookups.transform_df_uprn_lookup(uprn_lookup_df)
    domestic_uprns_df = domestic_uprns_df.join(
        uprn_lookup_df.select(["UPRN", "postcode", "country"]), how="left", on="UPRN"
    )
    del uprn_lookup_df

    # Get geopoints of UPRNs
    domestic_uprns_gdf = uprns.generate_gdf_uprn_coords(df=domestic_uprns_df)

    # ------------------------ #
    # IMPUTE PROPERTY TYPE FLAT
    # Create boolean column called `property_type_flat` to identify flats
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=domestic_uprns_gdf)
    features_df = domestic_uprns_df.with_columns(
        pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat")
    )

    # ------------------------ #
    # PREDICT BLOCK OF FLATS CLASSIFICATION
    # Load building footprint data
    # TODO scale beyond sampling areas
    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    all_uprns_df = load_geodata.load_df_osopen_uprn(
        grid_squares=local_authority_dict["grid_squares"]
    ).with_columns(
        pl.col("UPRN").is_in(domestic_uprns_df["UPRN"]).alias("is_domestic"),
        pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat"),
    )
    all_uprns_gdf = uprns.generate_gdf_uprn_coords(df=all_uprns_df)

    # Map UPRNs to the ID of the building they're in
    uprn_building_id_df = uprns.map_dict_uprns_to_building_id(
        uprns_gdf=all_uprns_gdf,
        buildings_gdf=buildings_gdf,
        id_col="ID",
        as_dict=False,
    )
    uprn_building_id_dict = dict(
        zip(uprn_building_id_df["UPRN"], uprn_building_id_df["ID"])
    )

    # Generate features for block of flats classification model
    building_features_df = feature_engineering.generate_df_features(
        buildings_gdf=buildings_gdf,
        uprns_gdf=all_uprns_gdf,
        mapping_df=uprn_building_id_df.to_pandas(),
        id_col="ID",
    )

    # TODO make this robust by automatically loading same labelled data used to train model
    labelled_df = pl.read_parquet(
        config["data"]["processed"]["manually_labelled_block_of_flats"].format(
            release_date=args.model_release
        )
    ).rename({"building_id": "ID"})

    # Load trained block of flats classifier model
    clf = base_getters.load_pickle(
        config["output"]["model"]["block_of_flats_model"].format(
            release_date=args.model_release
        )
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

    del (
        all_uprns_df,
        all_uprns_gdf,
        uprn_building_id_df,
        uprn_building_id_dict,
        building_features_df,
        labelled_df,
        clf,
    )

    # ------------------------ #
    # ESTIMATE OUTDOOR SPACE
    # TODO scale beyond Plymouth. This is a temporary fix to working with multiple LAs
    print("Loading land registry data...")
    inspire_file_gdf = gpd.read_parquet(
        config["data"]["processed"]["inspire_file_names"]
    )

    # Get file names of INSPIRE files which intersect with UPRNs
    inspire_file_names = inspire_file_gdf.sjoin(
        domestic_uprns_gdf, how="inner", predicate="intersects"
    )["inspire_file_name"].unique()

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
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    # Get intersection of building footprint polygons and land polygons
    intersection_gdf = outdoor_space.generate_gdf_building_intersections(
        land_parcels_gdf=land_parcels_gdf,
        buildings_gdf=buildings_gdf,
    )

    # Clip land parcels to barriers
    clipping_gdf = barriers.load_transform_gdf_polygon_barriers(
        grid_squares=local_authority_dict["grid_squares"]
    )
    land_parcels_gdf = outdoor_space.clip_gdf_land_parcels_to_barriers(
        land_parcels_gdf=land_parcels_gdf,
        intersection_gdf=intersection_gdf,
        barriers_gdf=clipping_gdf,
    )

    # Get outdoor space
    outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
        building_intersections_gdf=intersection_gdf, land_parcels_gdf=land_parcels_gdf
    )
    uprns_space_df = outdoor_space.sjoin_df_uprn_to_outdoor_space(
        uprns_gdf=domestic_uprns_gdf, outdoor_space_gdf=outdoor_space_gdf
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
    epc_df = load_data.load_df_domestic_epc(
        grid_squares=local_authority_dict["grid_squares"],
        columns=[
            "UPRN",
            "TENURE",
            "BUILT_FORM",
            "CURRENT_ENERGY_RATING",
            "SOLAR_WATER_HEATING_FLAG",
            "PHOTO_SUPPLY",
            "ENERGY_CONSUMPTION_CURRENT",
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
        ],
    )

    # Add listed building boolean flag

    # Load listed buildings geodataframe for relevant nations
    countries = features_df["country"].drop_nulls().unique()
    if len(countries) > 1:
        nation = "GB"
    else:
        mapping = {
            "E": "England",
            "S": "Scotland",
            "W": "Wales",
        }
        nation = mapping[countries[0]]
    # TODO this seems to have gotten slower after specifying nation over GB?
    listed_buildings_gdf = listed_buildings.transform_gdf_listed_buildings(
        nation=nation
    )

    features_df = listed_buildings.extend_df_listed_building_bool(
        features_df=features_df,
        uprns_gdf=domestic_uprns_gdf,
        buildings_gdf=buildings_gdf,
        listed_buildings_gdf=listed_buildings_gdf,
    )

    del listed_buildings_gdf

    # Add off-gas label

    off_gas_postcodes = off_gas.load_transform_list_off_gas_postcodes()
    features_df = features_df.with_columns(
        pl.col("postcode").is_in(off_gas_postcodes).alias("off_gas")
    )
    del off_gas_postcodes

    # Add distance to salt water

    # Load coastline and clip to LA boundaries first
    # Load boundary
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=local_authority_dict["valid_local_authorities"]
    )
    # Adding 10km of buffer first means that all the coastline within 1500m of every UPRN in the LA is loaded.
    # The extra distance means the clipped geometry extends far beyond the LA boundaries so we don't mislabel points
    # as being close to the coastline (i.e. because they are close to the buffer).
    boundary = boundary_gdf.geometry.buffer(10000).union_all()
    coast_gdf = load_geodata.load_gdf_gb_coast_boundaries(
        clip=boundary, bbox=boundary.bounds
    )

    features_df = coast.extend_df_near_coastline_bool(
        features_df=features_df,
        uprns_gdf=domestic_uprns_gdf,
        coast_gdf=coast_gdf,
        distance_threshold_m=config["constant"]["coastline"][
            "distance_from_coastline_threshold_m"
        ],
        simplify_tolerance_m=config["constant"]["coastline"]["simplify_tolerance_m"],
    )

    del coast_gdf, boundary_gdf

    # Add conservation area boolean flag
    uprns_protected_areas_df = protected_areas.load_transform_df_uprn_in_protected_area(
        gdf=domestic_uprns_gdf
    )

    features_df = protected_areas.extend_df_protected_area_bool(
        features_df=features_df,
        protected_areas_df=uprns_protected_areas_df,
    )

    del uprns_protected_areas_df

    # ------------------------ #
    # SAVE OUTPUTS

    if args.save:
        output_path = save_utils.get_str_output_path(
            "domestic_uprns_with_features",
            release_date=release_date,
            local_authority=local_authority_dict["url_slug"],
        )
        save_utils.save_to_s3(features_df, path=output_path)
        manifest_utils.generate_and_save_run_manifest_to_s3(
            output_path,
            stage="add_features",
            local_authority=local_authority_dict["url_slug"],
            row_count=len(features_df),
            params={
                "local_authorities": args.local_authorities,
                "release_date": release_date,
            },
        )
