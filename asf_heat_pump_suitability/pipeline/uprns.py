"""Step 1: Filter OS Open UPRNs to domestic (residential) UPRNs.

Reads the OS Open UPRN dataset from S3, filters to residential properties using
building footprints, EPC registers, and POI data, and writes the result to S3.

Run locally:
    uv run ahps-uprns [--area <area>]

Run on the cloud via arm_orbit:
    orbit launch --script asf_heat_pump_suitability/pipeline/uprns.py --team <team> --project <project>

Area options (--area flag):
    plymouth            Plymouth Local Authority only (default for dev/test)
    plymouth_similar    Plymouth + Liverpool, Portsmouth, Southampton, Swansea
    sampling            Plymouth + Bath, Bradford, Glasgow, Manchester, Nottingham
    gb                  Full Great Britain
"""

import argparse
import logging

import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata, load_tree_input
from asf_heat_pump_suitability.pipeline.transform import non_residential_entities, poi, uprns
from asf_heat_pump_suitability.utils import save_utils

logger = logging.getLogger(__name__)

AREA_CHOICES = ["plymouth", "plymouth_similar", "sampling", "gb"]


def parse_arguments() -> argparse.Namespace:
    """Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated Namespace.
    """
    parser = argparse.ArgumentParser(description="Filter OS Open UPRNs to domestic UPRNs.")
    parser.add_argument(
        "--area",
        help=(
            "Geographic area to filter UPRNs for. "
            "'plymouth' (default), 'plymouth_similar', 'sampling', or 'gb' (full Great Britain)."
        ),
        type=str,
        choices=AREA_CHOICES,
        default="plymouth",
    )
    return parser.parse_args()


def run(area: str = "plymouth") -> None:
    """Filter OS Open UPRNs to domestic UPRNs for a specified area and write to S3.

    Args:
        area: Geographic area identifier. One of 'plymouth', 'plymouth_similar',
            'sampling', or 'gb'.
    """
    output_path = config["output"]["residential_uprns_template"].format(area=area)

    # Load all UPRNs
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    if area == "plymouth":
        logger.info("Filtering to Plymouth Local Authority...")
        grid_squares = config["constant"]["grid_squares"]["plymouth"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(select_las="Plymouth")
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    elif area == "plymouth_similar":
        logger.info("Filtering to Plymouth + similar cities...")
        grid_squares = config["constant"]["grid_squares"]["plymouth_similar_cities"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=config["constant"]["plymouth_similar_cities"]
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    elif area == "sampling":
        logger.info("Filtering to sampling areas...")
        grid_squares = config["constant"]["grid_squares"]["sampling_areas"]
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            select_las=config["constant"]["sampling_areas"]
        )
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

    else:  # gb
        logger.info("Processing full Great Britain...")
        grid_squares = None

    poi_gdf = load_tree_input.load_gdf_poi()
    poi_gdf = poi.transform_gdf_poi(
        poi_gdf,
        filter_categories=poi.load_set_non_domestic_poi_categories(),
    )

    layers = {
        f"{layer}_gdf": load_tree_input.load_gdf_os_openmap_local_layer(layer=layer, grid_squares=grid_squares)
        for layer in ["important_building", "railway_station", "building"]
    }

    non_residential_buildings_gdf = non_residential_entities.generate_gdf_non_residential_buildings(
        **layers, poi_gdf=poi_gdf, uprns_gdf=uprns_gdf
    )

    residential_uprns_gdf = uprns.filter_gdf_residential_uprns(
        uprn_gdf=uprns_gdf,
        buildings_gdf=layers["building_gdf"],
        non_residential_buildings_gdf=non_residential_buildings_gdf,
    )

    keep_cols = ["UPRN", "X_COORDINATE", "Y_COORDINATE", "LATITUDE", "LONGITUDE"]
    if "LAD23CD" in residential_uprns_gdf.columns:
        keep_cols += ["LAD23CD", "LAD23NM"]

    df = pl.from_pandas(residential_uprns_gdf[keep_cols])
    save_utils.save_to_s3(df, output_path)
    logger.info(f"Saved {len(df)} residential UPRNs to {output_path}")


def main() -> None:
    """Entry point registered as the ``ahps-uprns`` console script."""
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    run(area=args.area)


if __name__ == "__main__":
    main()
