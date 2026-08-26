"""
Script to compute contextual information for clusters including:
- Proportion of attachment types, tenure types, EPC ratings of properties within clusters
- Median outdoor space of properties within clusters
- Whether any properties within clusters are in protected areas, off-gas, within 1500m of coastline
- Number of properties, number of properties in listed buildings and number of properties with solar PV

Run:
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities LOCAL_AUTHORITIES

Set `--detail "simplified"` to use simplified spatial signature polygons to label city centres. The default is "full" which uses the fully detailed spatial signatures framework.

Add --save to save the output to S3 as a geojson with geometry and contextual features per cluster.

Set --release_date to specify the YYYYMMDD dated release directory to read inputs from and
save outputs to. Defaults to running the pipeline using today's date. Multi-day runs
should pass the same --release_date to every stage.
"""

import argparse
import polars as pl
import geopandas as gpd
import numpy as np
import pandas as pd
import json
import os
from dotenv import load_dotenv

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.pipeline.cluster import cluster
from asf_heat_pump_suitability.getters import (
    load_boundaries,
)
from asf_heat_pump_suitability.pipeline.transform import city_centres

# Load environment variables from .env file
load_dotenv()

ANCHOR_LOAD_RADIUS = config["constant"]["anchor_radius"]
COASTLINE_DISTANCE_THRESHOLD_M = config["constant"]["coastline"][
    "distance_from_coastline_threshold_m"
]


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
        "--detail",
        help="Level of detail for spatial signatures dataset to label city centres. Takes values 'simplified' or 'full'. Defaults to 'full'.",
        required=False,
        default="full",
        type=str,
    )

    parser.add_argument(
        "--save",
        help="Whether to save the output to S3.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--release_date",
        help="Release date in YYYYMMDD format used for the dated input and output directories. Defaults to today's date.",
    )

    return parser.parse_args()


def extend_df_contextual_features(
    clusters_df: pl.DataFrame,
    uprns_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Extend clusters dataframe with contextual features by aggregating features at UPRN level to cluster level.
    Contextual features added include:
    - Attachment type proportions
    - tenure type proportions
    - EPC rating proportions
    - Median outdoor space
    - number of properties in listed buildings
    - number of off-gas properties
    - proximity to coastline flag (within 1500m)
    - protected area flag
    - within anchor load radius flag
    - Logic trace for each cluster explaining why the assigned technology was chosen

    Args:
        clusters_df (pl.DataFrame): dataframe of clusters with cluster_id and `within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load` feature
        uprns_df (pl.DataFrame): dataframe of UPRNs with cluster_id and relevant features for aggregation
    Returns:
        pl.DataFrame: dataframe with remaining features per cluster
    """

    dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]

    unknowns_mapping = {
        "": "unknown",
        "null": "unknown",
    }

    uprns_df = uprns_df.with_columns(
        pl.col(col)
        .cast(pl.String)
        .str.to_lowercase()
        .str.strip_chars()
        .replace(unknowns_mapping)
        .fill_null("unknown")
        for col in dummy_cols
    )

    # Get value counts per feature
    dummy_contextual_feat_df = (
        uprns_df.select(dummy_cols + ["cluster_id"])
        .to_dummies(columns=dummy_cols)
        .group_by("cluster_id")
        .sum()
    )

    # Keep only the columns that start with the dummy column prefixes, e.g. "ATTACHMENT_", "TENURE_", "CURRENT_ENERGY_RATING_"
    dummy_cols_to_keep = [
        col
        for col in dummy_contextual_feat_df.columns
        if any(col.lower().startswith(prefix.lower()) for prefix in dummy_cols)
    ]
    dummy_contextual_feat_df = dummy_contextual_feat_df.select(
        ["cluster_id"] + dummy_cols_to_keep
    )

    # lower case column names
    dummy_contextual_feat_df = dummy_contextual_feat_df.rename(
        {
            col: col.lower().replace(" ", "_").replace("-", "_")
            for col in dummy_contextual_feat_df.columns
        }
    )

    clusters_df = clusters_df.join(
        dummy_contextual_feat_df, how="left", on="cluster_id"
    )

    contextual_feat_clusters_df = (
        uprns_df.group_by("cluster_id")
        .agg(
            # n_uprns
            pl.col("UPRN").n_unique().alias("n_UPRNs"),
            # n_uprns_listed_building
            pl.col("in_listed_building").sum().alias("n_uprns_in_listed_building"),
            # n_uprns_missing_listed_building_flag
            pl.col("in_listed_building")
            .is_null()
            .sum()
            .alias("n_uprns_missing_listed_building_flag"),
            # n_uprns_solar_pv
            pl.col("has_solar_pv").sum().alias("n_uprns_solar_pv"),
            # n_uprns_missing_solar_pv_flag
            pl.col("has_solar_pv")
            .is_null()
            .sum()
            .alias("n_uprns_missing_solar_pv_flag"),
            # n_uprns_off_gas
            pl.col("off_gas").sum().alias("n_uprns_off_gas"),
            # n_uprns_missing_off_gas_flag
            pl.col("off_gas").is_null().sum().alias("n_uprns_missing_off_gas_flag"),
            # median estimated energy consumption in 12 months (in kWh/m2)
            pl.col("ENERGY_CONSUMPTION_CURRENT")
            .median()
            .alias("median_estimated_energy_consumption_12_months_kwh_per_m2"),
            # median_outdoor_space
            pl.col("max_contiguous_outdoor_space_area_m2")
            .median()
            .alias("median_outdoor_space_m2"),
            # near_coastline flag
            pl.when(
                pl.col(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline")
                .is_null()
                .all()
            )
            .then(pl.lit("Unknown"))
            .when(pl.col(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias(f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline"),
            # in_protected_area flag
            pl.when(pl.col("in_protected_area").is_null().all())
            .then(pl.lit("Unknown"))
            .when(pl.col("in_protected_area").any())
            .then(pl.lit("Yes"))
            .otherwise(pl.lit("No"))
            .alias("in_protected_area"),
            # Counts of UPRNs near salt water, and in protected areas
            pl.col("within_1500m_coastline")
            .sum()
            .alias("n_uprns_within_1500m_of_coastline"),
            pl.col("in_protected_area").sum().alias("n_uprns_in_protected_area"),
        )
        .select(
            [
                "cluster_id",
                "n_UPRNs",
                "n_uprns_in_listed_building",
                "n_uprns_missing_listed_building_flag",
                "n_uprns_solar_pv",
                "n_uprns_missing_solar_pv_flag",
                "n_uprns_off_gas",
                "n_uprns_missing_off_gas_flag",
                "median_estimated_energy_consumption_12_months_kwh_per_m2",
                "median_outdoor_space_m2",
                f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline",
                "in_protected_area",
                "n_uprns_within_1500m_of_coastline",
                "n_uprns_in_protected_area",
            ]
        )
    )

    clusters_df = clusters_df.join(
        contextual_feat_clusters_df, how="left", on="cluster_id"
    )

    # Switching from booleans to Yes/No/Unknown `within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load`
    clusters_df = clusters_df.with_columns(
        pl.when(pl.col(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load").is_null())
        .then(pl.lit("Unknown"))
        .when(pl.col(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load"))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias(f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load")
    )

    # Add percentages used for sorting & filtering in the tool
    # Tenure cols: owner occupied, social rented, private rented percentages
    tenure_cols = [
        col for col in dummy_contextual_feat_df.columns if col.startswith("tenure_")
    ]
    # Adding percentages of properties with solar PV and off-gas, which are also used for filtering in the tool
    perc_cols = tenure_cols + [
        "n_uprns_solar_pv",
        "n_uprns_missing_solar_pv_flag",
        "n_uprns_off_gas",
        "n_uprns_missing_off_gas_flag",
    ]
    clusters_df = clusters_df.with_columns(
        [
            (pl.col(col) / pl.col("n_UPRNs") * 100).alias(
                "perc_" + col if col in tenure_cols else "perc_" + col.split("n_")[1]
            )
            for col in perc_cols
        ]
    )

    return clusters_df


def create_gdf_contextual_features(
    uprns_df: pl.DataFrame,
    clusters_gdf: gpd.GeoDataFrame,
    hn_zones_gdf: gpd.GeoDataFrame,
    spatial_signatures_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Create geodataframe with cluster_id, geometry and contextual features for each cluster

    Args:
        uprns_df (pl.DataFrame): dataframe of UPRNs and UPRN-level features
        clusters_gdf (gpd.GeoDataFrame): geodataframe of clusters with geometry and cluster_id
        hn_zones_gdf (gpd.GeoDataFrame): geodataframe of heat network zones with geometry and source annotation
        city_centre_signatures_gdf (gpd.GeoDataFrame): geodataframe of spatial signatures representing city centre areas with geometry and source annotation
    Returns:
        gpd.GeoDataFrame: geodataframe with cluster_id, geometry and contextual features for each cluster (CRS: EPSG:4326)
    """

    clusters_with_contextual_features_df = extend_df_contextual_features(
        clusters_df=pl.from_pandas(
            clusters_gdf.drop(columns="geometry")
        ),  # drop geometry for now and use polars
        uprns_df=uprns_df,
    )

    # TODO identify source of empty clusters and fix - temporary fix to remove empty clusters
    clusters_with_contextual_features_df = clusters_with_contextual_features_df.filter(
        pl.col("n_UPRNs").is_not_null()
    )

    # Adding the geometry back to the clusters dataframe
    clusters_with_contextual_features_gdf = gpd.GeoDataFrame(
        clusters_with_contextual_features_df.to_pandas().merge(
            clusters_gdf[["cluster_id", "geometry"]],
            how="left",
            on="cluster_id",
        ),
        geometry="geometry",
        crs="EPSG:27700",
    )

    # Add in_hn_zone and in_city_centre flags to clusters_gdf
    clusters_with_contextual_features_gdf["in_hn_zone"] = (
        clusters_with_contextual_features_gdf.intersects(hn_zones_gdf.union_all())
    ).map({True: "Yes", False: "No"})
    clusters_with_contextual_features_gdf["in_city_centre"] = (
        clusters_with_contextual_features_gdf.intersects(
            spatial_signatures_gdf.union_all()
        )
    ).map({True: "Yes", False: "No"})

    # Add logic trace for each cluster explaining why the assigned technology was chosen
    assigned_tech = clusters_with_contextual_features_gdf["assigned_tech"]
    tech_types = config["constant"]["tech_types"]
    dhn_potential = (clusters_with_contextual_features_gdf["in_hn_zone"] == "Yes") | (
        clusters_with_contextual_features_gdf["in_city_centre"] == "Yes"
    )
    near_anchor_load = (
        clusters_with_contextual_features_gdf[
            f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load"
        ]
        == "Yes"
    )

    logic_trace = [
        # 1) Communal solution + near anchor load + DHN potential
        # TODO: remove sentence about blocks of flats once we separate clusters that are communal due to blocks of flats from those that are communal due to anchor load proximity
        (
            (assigned_tech == tech_types["communal"])
            & near_anchor_load
            & dhn_potential,
            f"This cluster:\n- is in an area of 'district heat network' potential,\n- contains homes that have no or little outdoor space (up to 30m2),\n and it is within {ANCHOR_LOAD_RADIUS}m of an anchor load (e.g. a hospital or school),\n so homes are most suitable for a 'district heat network' connection when/if a district heat network is constructed. \n A 'communal solution' could be considered as an alternative solution because the homes have no or little outdoor space (up to 30m2) and it is within {ANCHOR_LOAD_RADIUS}m of an anchor load (e.g. a hospital or school). This 'communal solution' could be integrated into a 'district heat newtork' in the future.\n There might also be one or multiple blocks of flats in the cluster.\n",
        ),
        # 2) Communal solution + near anchor load
        # TODO: remove sentence about blocks of flats once we separate clusters that are communal due to blocks of flats from those that are communal due to anchor load proximity
        (
            (assigned_tech == tech_types["communal"]) & near_anchor_load,
            f"This cluster is assigned 'communal solution' because the homes have no or little outdoor space (up to 30m2) and it is within {ANCHOR_LOAD_RADIUS}m of an anchor load (e.g. a hospital or school).\n There might also be one or multiple blocks of flats in the cluster.",
        ),
        # 3) Communal solution (blocks of flats)+ DHN potential
        (
            (assigned_tech == tech_types["communal"] & dhn_potential),
            "This cluster is in an area of 'district heat network' potential and it contains one or multiple blocks of flats, so homes are most suitable for a 'district heat network' connection when/if a district heat network is constructed.\n A ‘communal solution’ could also be considered as an alternative solution, because it contains one or multiple blocks of flats. This 'communal solution' could be integrated into a 'district heat network' in the future.",
        ),
        # 4) Communal solution (blocks of flats)
        (
            (assigned_tech == tech_types["communal"]),
            "This cluster is assigned 'communal solution' because it contains one or multiple blocks of flats.",
        ),
        # 5) Networked heat pump + DHN potential
        (
            (assigned_tech == tech_types["networked"]) & dhn_potential,
            "This cluster is in an area of 'district heat network' potential and homes have no or little outdoor space (up to 30m2), so homes are most suitable for a 'district heat network' connection when/if a district heat network is constructed.\n A 'networked heat pump' could be considered as an alternative solution because multiple properties within these buildings have no or little outdoor space (up to 30m2).\n",
        ),
        # 6) Networked heat pump
        (
            (assigned_tech == tech_types["networked"]),
            "This cluster is assigned 'networked heat pump' because multiple properties within these buildings have no or little outdoor space (up to 30m2)",
        ),
        # 7) Individual solution or networked HP (lack of outdoor space info) + DHN potential
        (
            (assigned_tech == tech_types["individual_or_networked"]) & dhn_potential,
            "Outdoor space is unknown for multiple properties in buildings within this cluster, so the model is unable to assign a technology group.\n If there is little (up to 30m2) contiguous outdoor space:\n -This cluster is in an area of 'district heat network' potential and if homes have no or little outdoor space (up to 30m2), then homes would be most suitable for a district heat network connection when/if a 'district heat network' is constructed.\n A 'networked heat pump' could also be considered as an alternative solution, because multiple properties within these buildings have no or little outdoor space (up to 30m2).\n\nIf there is sufficient outdoor space for each property (above 30m2):\n - 'Individual solutions' will be the most suitable option for properties in this cluster because outdoor space is above 30m2 for all properties.\n - The cluster is also in an area of 'district heat network' potential. If a district heat network is built, connection to the network could be offered.",
        ),
        # 8) Individual solution or networked HP (lack of outdoor space info)
        (
            (assigned_tech == tech_types["individual_or_networked"]),
            "Outdoor space is unknown for multiple properties in buildings within this cluster, so the model is unable to assign a technology group.\n If there is little (up to 30m2) contiguous outdoor space, a 'networked heat pump' solution is most suitable. If there is sufficient outdoor space for each property (above 30m2) an 'individual solution' will be most suitable.",
        ),
        # 9) Individual solution + DHN potential
        (
            (assigned_tech == tech_types["individual"]) & dhn_potential,
            "This cluster is assigned 'individual solution' because outdoor space is above 30m2 for all properties in this cluster.\nThe cluster is also in an area of 'district heat network' potential. If a district heat network is built, connection to the network could be offered.",
        ),
        # 10) Individual solution
        (
            (assigned_tech == tech_types["individual"]),
            "This cluster is assigned 'individual solution' because outdoor space is above 30m2 for all properties in this cluster.",
        ),
    ]

    conditions, logic = zip(*logic_trace)
    clusters_with_contextual_features_gdf["logic_trace"] = np.select(
        conditions, logic, default="Not accounted for in logic trace."
    )

    return clusters_with_contextual_features_gdf


def create_json_contextual_features_metadata(
    clusters_with_contextual_features_gdf: gpd.GeoDataFrame,
    local_authorities: str,
    release_date: str,
    optional_data_layers: dict = None,
) -> json:
    """
    Create json with cluster level data and associated metadata.

    Args:
        clusters_with_contextual_features_gdf (gpd.GeoDataFrame): geodataframe with cluster_id, geometry and contextual features for each cluster (CRS: EPSG:4326)
        local_authorities (str): local authority or authorities for which the data was generated
        release_date (str): release date in YYYYMMDD format of the dated release directory the data belongs to
        optional_data_layers (dict): dictionary of optional data layers with layer name as key and geodataframe as value

    Returns:
       json: geojson file with metadata in the `metadata` key and cluster level data in geojson format in the `features` key

    """

    print("Adding metadata and converting to geojson format...")
    # Convert to geojson format and add metadata
    clusters_json = json.loads(
        clusters_with_contextual_features_gdf.to_json(drop_id=True)
    )
    for feature in clusters_json["features"]:
        feature["properties"]["layer"] = "clusters_with_contextual_features"

    if optional_data_layers:
        for layer_name, layer_gdf in optional_data_layers.items():
            layer_json = json.loads(layer_gdf.to_json(drop_id=True))
            for feature in layer_json["features"]:
                feature["properties"]["layer"] = layer_name
            clusters_json["features"].extend(layer_json["features"])

    metadata = {
        "Release date": release_date,
        "Local authority": local_authorities,
    }

    # append metadata from config base.yaml
    metadata.update(config["metadata"])
    metadata["Variable names and descriptions"][
        f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline"
    ] = (
        metadata["Variable names and descriptions"]
        # Pop deletes the original key and returns the value
        .pop("within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline").format(
            COASTLINE_DISTANCE_THRESHOLD_M=COASTLINE_DISTANCE_THRESHOLD_M
        )
    )
    metadata["Variable names and descriptions"][
        f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load"
    ] = (
        metadata["Variable names and descriptions"]
        # Pop deletes the original key and returns the value
        .pop("within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load").format(
            ANCHOR_LOAD_RADIUS=ANCHOR_LOAD_RADIUS
        )
    )

    geojson_file = {
        "type": "FeatureCollection",
        "metadata": metadata,
        "features": clusters_json["features"],
    }

    return geojson_file


if __name__ == "__main__":
    from asf_heat_pump_suitability.getters import load_geodata
    from asf_heat_pump_suitability.pipeline.transform import local_authority
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()
    local_authorities = args.local_authorities
    detail_level = args.detail

    tolerance_m = config["constant"]["clustering"]["tolerance_m"]

    local_authority_dict = local_authority.get_dict_la_data(local_authorities)

    release_date = save_utils.get_str_release_date(args.release_date)

    print(f"Loading {local_authorities} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        save_utils.get_str_output_path(
            "domestic_uprns_with_features",
            release_date=release_date,
            check_exists=True,
            local_authority=local_authority_dict["url_slug"],
        )
    )

    buildings_gdf = load_geodata.load_gdf_os_openmap_layer(
        layer="building", grid_squares=local_authority_dict["grid_squares"]
    )

    print("Loading clusters...")
    clusters_gdf = gpd.read_parquet(
        save_utils.get_str_output_path(
            "tech_clusters",
            release_date=release_date,
            check_exists=True,
            local_authorities=local_authority_dict["url_slug"],
        ),
    ).to_crs(epsg=27700)

    uprns_df = cluster.map_df_uprns_to_clusters(
        uprns_df=uprns_df, buildings_gdf=buildings_gdf, clusters_gdf=clusters_gdf
    )

    print(
        "Loading local authority boundaries, heat network zones and spatial signatures..."
    )
    boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
        select_las=local_authority_dict["valid_local_authorities"]
    )
    # HN zones
    hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(boundary=boundary_gdf)

    # Spatial signatures for city centres
    spatial_signatures_gdf = load_geodata.load_gdf_spatial_signatures_gb(
        detail_level=detail_level,
        boundary=boundary_gdf,
        signature_types=city_centres.CITY_CENTRE_TYPES,
    )

    print("Computing contextual features for clusters...")
    clusters_with_contextual_features_gdf = create_gdf_contextual_features(
        uprns_df=uprns_df,
        clusters_gdf=clusters_gdf,
        hn_zones_gdf=hn_zones_gdf,
        spatial_signatures_gdf=spatial_signatures_gdf,
    )

    print("Simplifying geometries using tolerance_m...")
    clusters_with_contextual_features_gdf["geometry"] = (
        clusters_with_contextual_features_gdf["geometry"].simplify(
            tolerance=tolerance_m, preserve_topology=True
        )
    )

    print("Converting to EPSG:4326 for geojson output...")
    clusters_with_contextual_features_gdf = (
        clusters_with_contextual_features_gdf.to_crs(epsg=4326)
    )

    print("Creating layer with district HN potential and converting to EPSG:4326...")
    hn_potential = pd.concat(
        [
            hn_zones_gdf[["geometry", "source_annotation"]],
            # Create a single polygon for all spatial signatures to represent city centres
            gpd.GeoDataFrame(
                {
                    "source_annotation": [
                        spatial_signatures_gdf["source_annotation"].iloc[0]
                    ],
                    "geometry": [spatial_signatures_gdf.geometry.union_all()],
                },
                crs=spatial_signatures_gdf.crs,
            ),
        ]
    ).to_crs(epsg=4326)

    print("Loading anchor property geodataframes and transforming to EPSG:4326...")
    combined_anchor_gdf = cluster.load_transform_anchor_property_gdfs(
        buildings_gdf=buildings_gdf, grid_squares=local_authority_dict["grid_squares"]
    )[["geometry"]]

    combined_anchor_gdf = combined_anchor_gdf[
        combined_anchor_gdf["geometry"].intersects(boundary_gdf.union_all())
    ].to_crs(epsg=4326)

    print("Loading ward boundaries and transforming to EPSG:4326...")
    ward_boundaries_gdf = load_boundaries.load_gdf_ward_boundaries(
        select_las=local_authority_dict["valid_local_authorities"],
        la_boundaries_gdf=boundary_gdf,
    )[["geometry"]].to_crs(epsg=4326)

    optional_data_layers = {
        "ward_boundaries": ward_boundaries_gdf,
        "anchor_loads": combined_anchor_gdf,
        "areas_of_district_heat_network_potential": hn_potential,
    }

    print("Creating json with contextual features for each cluster and metadata...")
    geojson_file = create_json_contextual_features_metadata(
        clusters_with_contextual_features_gdf=clusters_with_contextual_features_gdf,
        optional_data_layers=optional_data_layers,
        local_authorities=local_authorities,
        release_date=release_date,
    )

    if args.save:
        print("Saving geojson to S3... ")
        # Save to S3 as geojson
        s3_file_path = save_utils.get_str_output_path(
            "clusters_tech_contextual_info",
            release_date=release_date,
            local_authorities=local_authority_dict["url_slug"],
            tolerance_m=tolerance_m,
        )

        # Save to data science S3 bucket
        save_utils.save_to_s3(
            geojson_file,
            s3_file_path,
        )

        # Save to front-end S3 bucket for use in the tool
        front_end_staging_s3_path = os.environ.get("front_end_staging_s3_path")
        front_end_s3_bucket = os.environ.get("front_end_s3_bucket")
        file_name = s3_file_path.split("/")[-1]
        save_utils.save_to_s3(
            geojson_file,
            os.path.join(
                "s3://", front_end_s3_bucket, front_end_staging_s3_path, file_name
            ),
        )
