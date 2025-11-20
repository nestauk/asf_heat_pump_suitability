"""
Script to label UPRNs with flat / apartment property type boolean flag.

Run:
python asf_heat_pump_suitability/pipeline/run/flat_blocks.py --uprns path/to/domestic/UPRNs
"""

import argparse


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

    return parser.parse_args()


if __name__ == "__main__":
    import os
    import polars as pl

    from asf_heat_pump_suitability.utils import save_utils
    from asf_heat_pump_suitability.pipeline.impute import property_type
    from asf_heat_pump_suitability.pipeline.transform import uprns

    args = parse_arguments()

    # Load UPRN data
    print(f"Loading domestic UPRNs from: {args.uprns}")
    uprns_df = pl.read_parquet(
        args.uprns, columns=["UPRN", "X_COORDINATE", "Y_COORDINATE"]
    )

    # Get geopoints of UPRNs
    uprns_gdf = uprns.generate_gdf_uprn_coords(df=uprns_df)

    # Create boolean column called `property_type_flat` to identify flats
    flat_uprns = property_type.impute_set_flat_properties(uprns_gdf=uprns_gdf)
    uprns_df = uprns_df.with_columns(
        pl.col("UPRN").is_in(flat_uprns).alias("property_type_flat")
    )

    save_utils.save_to_s3(
        uprns_df,
        path=f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{os.path.basename(args.uprns).split('.')[0]}_with_flats.parquet",
    )
