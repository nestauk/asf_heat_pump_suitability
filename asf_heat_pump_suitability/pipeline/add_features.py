"""Step 2: Add features to domestic UPRNs.

Reads the domestic UPRN dataset produced by pipeline/uprns.py from S3, adds
geospatial features (flat/apartment flag, outdoor space estimates), and writes
the result to S3.

Run locally:
    python pipeline/add_features.py --uprns <s3-path-or-local-path>

Run on the cloud via arm_orbit:
    orbit launch --script pipeline/add_features.py --team <team> --project <project>
"""

import argparse
import logging
import os

import geopandas as gpd
import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.config.settings import load_settings
from asf_heat_pump_suitability.getters import load_tree_input
from asf_heat_pump_suitability.pipeline.impute import property_type
from asf_heat_pump_suitability.pipeline.transform import outdoor_space, uprns
from asf_heat_pump_suitability.pipeline.transform.uprns import get_area_config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.utils.storage import get_path, mock_aws_if_local

logger = logging.getLogger(__name__)


AREA_CHOICES = ["plymouth", "plymouth_similar", "sampling", "gb"]


def parse_arguments() -> argparse.Namespace:
    """Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated Namespace.
    """
    parser = argparse.ArgumentParser(description="Add features to domestic UPRNs.")
    parser.add_argument(
        "--uprns",
        help="S3 URI or local path to the domestic UPRN parquet file produced by pipeline/uprns.py.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--area",
        help=(
            "Geographic area the UPRNs were filtered to. Used to look up the INSPIRE land-registry "
            "parcel file and OS OpenMap Local grid squares in config. "
            "'plymouth' (default), 'plymouth_similar', 'sampling', or 'gb'."
        ),
        type=str,
        choices=AREA_CHOICES,
        default="plymouth",
    )
    return parser.parse_args()


def run(uprns_path: str, area: str = "plymouth") -> None:
    """Add features to domestic UPRNs and write enriched dataset to S3.

    Args:
        uprns_path: Path (S3 URI or local) to the domestic UPRNs parquet file.
        area: Geographic area identifier used to look up INSPIRE land-registry parcels
            and OS OpenMap Local grid squares from config. One of 'plymouth',
            'plymouth_similar', 'sampling', or 'gb'.
    """
    settings = load_settings()
    output_stem = os.path.basename(uprns_path).split(".")[0]
    s3_output = settings.output.features_template.format(uprns_stem=output_stem)
    output_path = get_path(s3_output, settings)

    # Derive grid squares and INSPIRE path from area config
    grid_squares, _ = get_area_config(area)
    inspire_path = config["data"]["geodata"]["inspire_land_registry"].get(area)

    # Load UPRN data
    logger.info(f"Loading domestic UPRNs from: {uprns_path}")
    uprns_df = pl.read_parquet(uprns_path, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"])

    # Convert to GeoDataFrame with BNG point geometries
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # ── Flat / apartment imputation ──────────────────────────────────────────
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    features_df = uprns_df.with_columns(pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat"))

    # ── Outdoor space estimation ─────────────────────────────────────────────
    if inspire_path is None:
        logger.warning(
            f"No INSPIRE land-registry data configured for area {area!r}; "
            "outdoor space features will be null for all UPRNs."
        )
    else:
        logger.info(f"Loading land registry data from: {inspire_path}")
        land_parcels_gdf = gpd.read_file(inspire_path)
        building_footprints_gdf = load_tree_input.load_gdf_os_openmap_local_layer(
            layer="building", grid_squares=grid_squares
        )

        intersection_gdf = outdoor_space.generate_gdf_building_intersections(
            land_parcels_gdf=land_parcels_gdf,
            building_footprints_gdf=building_footprints_gdf,
        )
        outdoor_space_gdf = outdoor_space.generate_gdf_outdoor_space(
            building_intersections_gdf=intersection_gdf,
            land_parcels_gdf=land_parcels_gdf,
        )
        uprns_space_df = outdoor_space.sjoin_df_uprn_to_outdoor_space(
            uprns_gdf=uprns_gdf,
            outdoor_space_gdf=outdoor_space_gdf,
        )
        uprns_space_df = outdoor_space.deduplicate_df_outdoor_space(uprns_space_df)

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

    # ── Save outputs ─────────────────────────────────────────────────────────
    save_utils.save_to_s3(features_df, output_path)
    logger.info(f"Saved features for {len(features_df)} UPRNs to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    with mock_aws_if_local():
        run(uprns_path=args.uprns, area=args.area)
