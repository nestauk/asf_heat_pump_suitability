"""
Contains script to add boolean flags to label residential UPRNs indicating whether they are:
- located within a heat network zone
- located within a city centre

To run the script:
python asf_heat_pump_suitability/pipeline/run/heat_network_city_centres.py

Set the optional `local_authorities` parameter to `plymouth`, `plymouth_similar`, or `sampling_areas`.
Set to `plymouth` to run for Plymouth Local Authority; `plymouth_similar` to run for Plymouth plus four other similar
Local Authorities (Liverpool, Portsmouth, Southampton, Swansea); 'sampling_areas' to run for Plymouth plus five other
Local Authorities for sampling buildings (Bath, Bradford, Glasgow, Manchester, Nottingham); or do not use to run for all
of Great Britain.
"""

import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    # Placeholder - this arg is to look up correct paths for residential UPRNs and heat network dataset
    parser.add_argument(
        "--local_authorities",
        help="Run script for either all of Great Britain; Plymouth only {plymouth}; or Plymouth and 4 similar local authorities {plymouth_similar}; or Plymouth and 5 different local authorities {sampling_areas}. Default to all of GB",
        type=str,
        default="GB",
        required=False,
    )

    parser.add_argument(
        "--uprns_path",
        help="Path to residential UPRN dataset with X and Y coordinates in parquet.",
        type=str,
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import base_getters, load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns
    from asf_heat_pump_suitability.pipeline.transform import (
        heat_network_zones,
        city_centres,
    )
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    las = args.local_authorities.lower()
    uprns_path = args.uprns_path

    if las == "gb":  # all GB
        # Placeholder for future implementation
        raise NotImplementedError("Processing for all of GB is not yet supported.")
    else:
        # Load residential OS UPRNs in Plymouth
        print(f"Loading residential UPRN dataset for {las}...")
        uprn_df = base_getters.load_df_from_s3(uprns_path)
        uprn_gdf = uprns.generate_gdf_uprn_coords(uprn_df)

        """ Existing heat network zones """

        # Load Plymouth existing heat network zone polygons
        la_names = config["constant"][las]["la_names"]
        print(f"Loading heat network zone data for {las} Local Authority...")
        hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(local_authority=las)

        print(hn_zones_gdf.head())

        # Label UPRNs in existing, potential and planned heat network zones
        print(f"Identifying residential UPRNs in heat network zones for {las}...")
        id_col = [col for col in hn_zones_gdf.columns if "ID" in col][0]

        hn_zone_uprn_df = heat_network_zones.label_gdf_heat_network_zone_uprns(
            uprn_gdf=uprn_gdf,
            hn_zone_gdf=hn_zones_gdf,
            usecols=[
                id_col,  # zone unique identifier
            ],
        )

        # Clean up columns
        hn_zone_uprn_df = hn_zone_uprn_df.select(
            ["UPRN", "LAD23NM", "X_COORDINATE", "Y_COORDINATE", "in_hn_zone", id_col]
        ).rename({id_col: "HNZoneID"})

        """ City centre areas """

        # Load spatial signature polygons
        print("Loading spatial signatures dataset...")
        spatial_signatures_gb_simplified_gdf = (
            load_geodata.load_gdf_spatial_signatures_gb(detail_level="full")
        )

        # Label UPRNs in city centres
        print(f"Identifying residential UPRNs in city centre areas for {las}...")
        hn_zone_uprn_gdf = uprns.generate_gdf_uprn_coords(hn_zone_uprn_df)
        hn_zone_city_centre_uprn_df = city_centres.label_gdf_city_centre_spatial_signatures_uprns(
            uprn_gdf=hn_zone_uprn_gdf,  # add city centre labels to gdf with hnz labels
            spatial_signatures_gdf=spatial_signatures_gb_simplified_gdf,
        )

        # Clean up columns
        hn_zone_city_centre_uprn_df = hn_zone_city_centre_uprn_df.select(
            [
                "UPRN",
                "LAD23NM",
                "X_COORDINATE",
                "Y_COORDINATE",
                # "HNZoneID",
                "in_hn_zone",
                "spatial_signature_types",
                "in_city_centre",
            ]
        )

    # Save residential UPRNs with existing heat network zone and city centre labels to S3
    save_utils.save_to_s3(
        hn_zone_city_centre_uprn_df,
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{args.local_authorities}_residential_uprns_with_hn_zones_city_centres.parquet",
    )
