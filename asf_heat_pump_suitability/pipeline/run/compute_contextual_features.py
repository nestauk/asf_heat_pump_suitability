"""
Script to compute contextual information for opportunity areas/clusters, e.g. property type, tenure, EPC rating, etc.

Run:
python asf_heat_pump_suitability/pipeline/run/compute_contextual_features.py --local_authorities LOCAL_AUTHORITIES

LOCAL_AUTHORITIES should be one of the options specified in base.yaml's `constant` section, e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.

Add --save to save the output to S3 as a geojson with geometry and contextual features per opportunity area/cluster.
"""

import argparse
import polars as pl
import geopandas as gpd


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
        "--save",
        help="Whether to save the output to S3.",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def filter_df_uprns_to_opportunity_areas(
    uprns_gdf: gpd.GeoDataFrame, areas_gdf: gpd.GeoDataFrame
) -> pl.DataFrame:
    """
    Filter UPRN geodataframe to only those which are within opportunity areas.

    Args:
        uprns_gdf (gpd.GeoDataFrame): geodataframe of UPRNs with geometry and EPC features
        areas_gdf (gpd.GeoDataFrame): geodataframe of opportunity areas with geometry and cluster_id
    Returns:
        pl.DataFrame: filtered UPRN dataframe with only UPRNs within opportunity areas
    """
    opportunity_areas_df = pl.from_pandas(
        uprns_gdf.sjoin(
            areas_gdf[["cluster_id", "geometry"]], how="right", predicate="within"
        ).drop(columns=["index_left", "geometry"])
    )

    return opportunity_areas_df


def extend_df_contextual_features(
    opportunity_areas_df: pl.DataFrame,
    uprns_gdf: gpd.GeoDataFrame,
) -> pl.DataFrame:
    """
    Extend opportunity areas dataframe with contextual features from UPRN geodataframe, including:
    - property type (including flats)
    - tenure
    - EPC rating
    - Median garden size
    - HN zone flag
    - City centre flag
    - number of listed buildings
    - number of off-gas properties
    - proximity to coastline flag
    - proximity to anchor load flag
    - conservation area flag

    Args:
        opportunity_areas_df (pl.DataFrame): dataframe with value counts per opportunity area for relevant features
        uprns_gdf (gpd.GeoDataFrame): geodataframe of UPRNs with geometry and relevant features
    Returns:
        pl.DataFrame: dataframe with remaining features per opportunity area
    """

    # Get the cluster_id for each UPRN by spatially joining UPRN geodataframe with opportunity area geodataframe
    uprns_df = pl.from_pandas(
        uprns_gdf.sjoin(
            areas_gdf[["cluster_id", "geometry"]], how="left", predicate="within"
        ).drop(columns=["geometry"])
    )

    dummy_cols = ["ATTACHMENT", "TENURE", "CURRENT_ENERGY_RATING"]
    # Get value counts per feature
    dummy_contextual_feat_df = (
        uprns_df.select(dummy_cols + ["cluster_id", "UPRN"])
        .to_dummies(columns=dummy_cols)
        .group_by("cluster_id")
        .agg(pl.all().sum())
        .drop("UPRN")
    )

    opportunity_areas_df = opportunity_areas_df.join(
        dummy_contextual_feat_df, how="left", on="cluster_id"
    )

    contextual_feat_clusters_df = uprns_df.group_by("cluster_id").agg(
        # median_outdoor_space
        pl.col("max_contiguous_outdoor_space_area_m2")
        .median()
        .alias("median_outdoor_space_m2"),
        # in_hn_zone flag
        pl.when(pl.col("in_hn_zone").any())
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("in_hn_zone"),
        # in_city_centre flag
        pl.when(pl.col("in_city_centre").any())
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("in_city_centre"),
        # n_uprns
        pl.col("UPRN").count().alias("n_UPRNs"),
        # n_uprns_listed_building
        pl.col("in_listed_building").count().alias("n_uprns_listed_building"),
        # n_uprns_off_gas
        pl.col("off_gas").count().alias("n_uprns_off_gas"),
        # near_coastline flag
        pl.col("near_coastline").any().alias("near_coastline"),
        # near_anchor_load flag
        pl.col("near_anchor_load").any().alias("near_anchor_load"),
        # in_conservation_area flag
        pl.col("in_conservation_area").any().alias("in_conservation_area"),
    )

    opportunity_areas_df = opportunity_areas_df.join(
        contextual_feat_clusters_df, how="left", on="cluster_id"
    )

    return opportunity_areas_df


if __name__ == "__main__":
    from asf_heat_pump_suitability.pipeline.transform import uprns

    args = parse_arguments()
    local_authorities = args.local_authorities

    print(f"Loading {local_authorities} domestic UPRNs...")
    uprns_df = pl.read_parquet(
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_with_features.parquet"
    )
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df).to_crs(epsg=27700)

    print("Loading opportunity areas...")
    areas_gdf = gpd.read_file(
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_tech_polygons_with_clusterID.geojson"
    ).to_crs(epsg=27700)

    print("Filtering to opportunity areas...")
    opportunity_areas_df = filter_df_uprns_to_opportunity_areas(
        uprns_gdf=uprns_gdf, areas_gdf=areas_gdf
    )

    print("Calculate remaining features per opportunity area...")
    opportunity_areas_df = extend_df_contextual_features(
        opportunity_areas_df=opportunity_areas_df, uprns_gdf=uprns_gdf
    )

    print("Remove clusters without any UPRNs within them...")
    opportunity_areas_df = opportunity_areas_df.filter(pl.col("n_UPRNs") > 0)

    if args.save:
        opportunity_areas_df = opportunity_areas_df.to_pandas().merge(
            areas_gdf[["cluster_id", "geometry"]],
            how="left",
            on="cluster_id",
        )

        # Saving as EPSG:4326 because we need lat/long for visualisation
        opportunity_areas_df = gpd.GeoDataFrame(
            opportunity_areas_df, geometry="geometry", crs="EPSG:27700"
        ).to_crs(epsg=4326)

        opportunity_areas_df.to_file(
            f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{local_authorities}/{local_authorities}_cluster_contextual_features.geojson",
            driver="GeoJSON",
        )
