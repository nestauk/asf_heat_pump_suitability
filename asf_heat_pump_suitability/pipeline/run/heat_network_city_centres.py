"""
Contains script to add boolean flags to label residential UPRNs indicating whether they are:
- located within a heat network zone
- located within a city centre (according to Spatial Signatures Framework)

To run the script:
python asf_heat_pump_suitability/pipeline/run/heat_network_city_centres.py --uprns_path path/to/residential/uprns.parquet

where you should replace `path/to/residential/uprns.parquet` with the path to the parquet file containing residential UPRNs with X and Y coordinates.

Set the `local_authorities` parameter to:
- `plymouth` for Plymouth only
- `plymouth_similar` for Plymouth and 4 similar local authorities (Liverpool, Portsmouth, Southampton, Swansea)
- `sampling_areas` for Plymouth and 5 different local authorities for sampling buildings (Bath, Bradford, Glasgow, Manchester, Nottingham)
- `greater_manchester_las` for all Greater Manchester local authorities (Bolton, Bury, Manchester, Oldham, Rochdale, Salford, Stockport, Tameside, Trafford, Wigan)
Defaults to `GB` (all of Great Britain), but this is not yet implemented.

Temporary (before we scale): Set up a new local authority or group of local authorities by adding an entry to the `constant` section of the config.yaml file.

Set --save to save the outputs to S3. By default, outputs are not saved.
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
        help="Local authority or authorities. See base.yaml's `constant` section for options e.g. `plymouth`, `plymouth_similar_cities`, `sampling_areas`, `greater_manchester_las`.",
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

    parser.add_argument(
        "--save",
        help="If --save is set, it saves outputs to S3.",
        type=bool,
        required=False,
        action="store_true",
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

    # TODO scale to GB
    if las == "gb":  # all GB
        # Placeholder for future implementation
        raise NotImplementedError("Processing for all of GB is not yet supported.")
    else:
        # Load residential OS UPRNs in Plymouth
        print(f"Loading residential UPRN dataset for {las}...")
        uprn_df = base_getters.load_df_from_s3(uprns_path)
        uprn_gdf = uprns.generate_gdf_uprn_coords(uprn_df)

        ### Existing heat network zones

        # Load Plymouth existing heat network zone polygons
        print(f"Loading heat network zone data for {las} Local Authority...")
        hn_zones_gdf = load_geodata.load_gdf_heat_network_zones(local_authority=las)

        # Label UPRNs in existing, potential and planned heat network zones
        print(f"Identifying residential UPRNs in heat network zones for {las}...")
        id_col = [col for col in hn_zones_gdf.columns if "ID" in col][0]
        print(f"Using Heat Network Zone {id_col} column as ID")

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

        ### City centre areas

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
                "in_hn_zone",
                "spatial_signature_types",
                "in_city_centre",
            ]
        )

    # Save residential UPRNs with existing heat network zone and city centre labels to S3
    if args.save:
        save_utils.save_to_s3(
            hn_zone_city_centre_uprn_df,
            f"s3://asf-heat-pump-suitability/local_heat_planning/outputs/{args.local_authorities}_residential_uprns_with_hn_zones_city_centres.parquet",
        )
