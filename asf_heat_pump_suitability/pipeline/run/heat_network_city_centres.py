"""
Contains script to add boolean flags to label residential UPRNs indicating whether they are:
- located within a heat network zone
- located within a city centre

To run the script:
python asf_heat_pump_suitability/pipeline/run/heat_network_city_centres.py

Set the optional `--area` parameter to `plymouth` (default), `plymouth_similar`, `sampling`,
or `gb` to select the geographic scope.  Heat network zone data is currently only available
for Plymouth; UPRNs in other areas will receive ``in_hn_zone = False``.
"""

import argparse
import logging

logger = logging.getLogger(__name__)

AREA_CHOICES = ["plymouth", "plymouth_similar", "sampling", "gb"]


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--area",
        help=(
            "Geographic area to label UPRNs for. "
            "'plymouth' (default), 'plymouth_similar', 'sampling', or 'gb' (full Great Britain). "
            "Heat network zone data is only available for 'plymouth'; other areas will have "
            "in_hn_zone=False for all UPRNs."
        ),
        type=str,
        choices=AREA_CHOICES,
        default="plymouth",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    import geopandas as gpd

    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import base_getters, load_geodata
    from asf_heat_pump_suitability.pipeline.transform import city_centres, heat_network_zones, uprns
    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.utils.storage import mock_aws_if_local

    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    area = args.area

    with mock_aws_if_local():
        # Load residential UPRNs for the specified area using the shared path template
        uprns_path = config["output"]["residential_uprns_template"].format(area=area)
        logger.info(f"Loading residential UPRN dataset for area: {area!r}...")
        uprn_df = base_getters.load_df_from_s3(uprns_path)
        uprn_gdf = uprns.generate_gdf_uprn_coords(uprn_df)

        # Load heat network zones for the area if available; fall back to empty GeoDataFrame
        # so that all UPRNs receive in_hn_zone=False without raising an error.
        try:
            hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(local_authority=area)
            logger.info(f"Loaded heat network zone data for {area!r}")
        except ValueError:
            logger.info(f"No heat network zone data available for {area!r}; all UPRNs will have in_hn_zone=False")
            hn_zones_gdf = gpd.GeoDataFrame(
                columns=["geometry", "ZoneID"],
                geometry="geometry",
                crs=config["constant"]["target_crs"],
            )

        # Label UPRNs in heat network zones
        hn_zone_uprn_df = heat_network_zones.label_gdf_heat_network_zone_uprns(
            uprn_gdf=uprn_gdf,
            hn_zone_gdf=hn_zones_gdf,
            usecols=["ZoneID"],
        )

        # Keep LAD columns only when they exist (not present for gb area)
        base_cols = ["UPRN", "X_COORDINATE", "Y_COORDINATE", "in_hn_zone", "ZoneID"]
        if "LAD23NM" in hn_zone_uprn_df.columns:
            base_cols = ["UPRN", "LAD23NM", "X_COORDINATE", "Y_COORDINATE", "in_hn_zone", "ZoneID"]
        hn_zone_uprn_df = hn_zone_uprn_df.select(base_cols).rename({"ZoneID": "HNZoneID"})

        # Load spatial signatures (GB-scale, no area filtering needed)
        logger.info("Loading spatial signatures dataset...")
        spatial_signatures_gb_gdf = load_geodata.load_gdf_spatial_signatures_gb(detail_level="full")

        # Label UPRNs in city centres
        logger.info(f"Identifying residential UPRNs in city centre areas for {area!r}...")
        hn_zone_uprn_gdf = uprns.generate_gdf_uprn_coords(hn_zone_uprn_df)
        hn_zone_city_centre_uprn_df = city_centres.label_gdf_city_centre_spatial_signatures_uprns(
            uprn_gdf=hn_zone_uprn_gdf,
            spatial_signatures_gdf=spatial_signatures_gb_gdf,
        )

        # Final column selection
        final_cols = [
            "UPRN",
            "X_COORDINATE",
            "Y_COORDINATE",
            "HNZoneID",
            "in_hn_zone",
            "spatial_signature_types",
            "in_city_centre",
        ]
        if "LAD23NM" in hn_zone_city_centre_uprn_df.columns:
            final_cols = [
                "UPRN",
                "LAD23NM",
                "X_COORDINATE",
                "Y_COORDINATE",
                "HNZoneID",
                "in_hn_zone",
                "spatial_signature_types",
                "in_city_centre",
            ]
        hn_zone_city_centre_uprn_df = hn_zone_city_centre_uprn_df.select(final_cols)

        # Save residential UPRNs with heat network zone and city centre labels to S3
        output_path = f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{area}_residential_uprns_with_hn_zones_city_centres.parquet"
        save_utils.save_to_s3(hn_zone_city_centre_uprn_df, output_path)
        logger.info(f"Saved output to {output_path}")
