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

    return parser.parse_args()


if __name__ == "__main__":
    from asf_heat_pump_suitability import config
    from asf_heat_pump_suitability.getters import base_getters, load_geodata
    from asf_heat_pump_suitability.pipeline.transform import uprns
    from asf_heat_pump_suitability.pipeline.transform import heat_network_zones
    from asf_heat_pump_suitability.utils import save_utils

    args = parse_arguments()

    if args.local_authorities.lower() == "plymouth":

        # Load residential OS UPRNs in Plymouth
        print("Loading residential UPRN dataset for Plymouth Local Authority...")
        uprn_df = base_getters.load_df_from_s3(
            config["data"]["processed"]["plymouth_residential_uprns"]
        )
        uprn_gdf = uprns.generate_gdf_uprn_coords(uprn_df)

        # Load Plymouth existing heat network zone polygons
        print("Loading heat network zone data for Plymouth Local Authority...")
        plymouth_hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(
            local_authority="plymouth"
        )

        # Filter for UPRNs in existing heat network zones
        print(
            "Identifying residential UPRNs in heat network zones for Plymouth Local Authority..."
        )
        hn_zone_uprn_df = heat_network_zones.filter_gdf_heat_network_zone_uprns(
            uprn_gdf=uprn_gdf,
            hn_zone_gdf=plymouth_hn_zones_gdf,
            usecols=[
                "ZoneID",  # zone unique identifier
            ],
        )

        # Clean up columns
        hn_zone_uprn_df = hn_zone_uprn_df[
            ["UPRN", "LAD23NM", "X_COORDINATE", "Y_COORDINATE", "in_hn_zone", "ZoneID"]
        ]
        hn_zone_uprn_df = hn_zone_uprn_df.rename({"ZoneID": "HNZoneID"})

        # TODO filter and add label for UPRNs in city centres

    elif args.local_authorities.lower() in ["plymouth_similar", "sampling_areas"]:
        # Placeholder for future implementation (subject to heat network zone data availability)
        raise NotImplementedError(
            f"Processing for {args.local_authorities} is not yet supported."
        )

    else:  # all GB
        # Placeholder for future implementation
        raise NotImplementedError("Processing for all of GB is not yet supported.")

    # Save residential UPRNs with labels to S3
    save_utils.save_to_s3(
        hn_zone_uprn_df,
        f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{args.local_authorities}_residential_uprns_with_hn_zones.parquet",
    )
