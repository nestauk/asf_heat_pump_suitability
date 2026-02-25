"""Step 1: Filter OS Open UPRNs to domestic (residential) UPRNs.

Reads the OS Open UPRN dataset from S3, filters to residential properties using
building footprints, EPC registers, and POI data, and writes the result to S3.

Run locally:
    python pipeline/uprns.py [--area <area>]

Run on the cloud via arm_orbit:
    orbit launch --script pipeline/uprns.py --team <team> --project <project>

Area options (--area flag):
    plymouth            Plymouth Local Authority only (default for dev/test)
    plymouth_similar    Plymouth + Liverpool, Portsmouth, Southampton, Swansea
    sampling            Plymouth + Bath, Bradford, Glasgow, Manchester, Nottingham
    gb                  Full Great Britain

Interactive development (VS Code / IPython):
    from asf_heat_pump_suitability.pipeline.uprns import run
    run(area="plymouth")
    # or: args = parse_arguments(["--area", "plymouth"]); run(**vars(args))
"""

import argparse
import logging

import polars as pl

from asf_heat_pump_suitability.config.settings import load_settings
from asf_heat_pump_suitability.getters import load_boundaries, load_geodata, load_tree_input
from asf_heat_pump_suitability.pipeline.transform import non_residential_entities, poi, uprns
from asf_heat_pump_suitability.pipeline.transform.uprns import get_area_config
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability.utils.storage import get_path, mock_aws_if_local

logger = logging.getLogger(__name__)

AREA_CHOICES = ["plymouth", "plymouth_similar", "sampling", "gb"]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Create ArgumentParser and parse.

    Args:
        argv: Argument list to parse. ``None`` reads from ``sys.argv`` (normal CLI
            behaviour). Pass an explicit list (e.g. ``[]`` or ``["--area", "gb"]``)
            for interactive / REPL use.

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
    return parser.parse_args(argv)


def run(area: str = "plymouth") -> None:
    """Filter OS Open UPRNs to domestic UPRNs for a specified area and write to S3.

    Args:
        area: Geographic area identifier. One of 'plymouth', 'plymouth_similar',
            'sampling', or 'gb'.
    """
    settings = load_settings()
    s3_output = settings.output.residential_uprns_template.format(area=area)
    output_path = get_path(s3_output, settings)

    # Load all UPRNs
    uprns_df = load_geodata.load_df_osopen_uprn()
    uprns_gdf = uprns.generate_gdf_uprn_coords(uprns_df)

    grid_squares, la_names = get_area_config(area)
    logger.info(f"Processing area: {area!r}")

    if la_names is not None:
        la_boundaries_gdf = load_boundaries.load_gdf_local_authority_boundaries(select_las=la_names)
        uprns_gdf = uprns_gdf.sjoin(
            la_boundaries_gdf[["LAD23CD", "LAD23NM", "geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns="index_right")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    with mock_aws_if_local():
        run(area=args.area)
