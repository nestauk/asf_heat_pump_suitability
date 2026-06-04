import argparse
import geopandas as gpd
from pathlib import Path
from asf_heat_pump_suitability.getters import load_boundaries, base_getters
from asf_heat_pump_suitability.utils import save_utils
from asf_heat_pump_suitability import config
import os
import polars as pl
import pandas as pd
import boto3


def get_boundary(local_authorities=None, boundary_file=None):
    """
    Finds boundary geometry given an input file containing a GeoDataFrame or Local Authority name.

    Args:
        local_authorities (str | list[str]): Local Authority or list of Local Authorities to find the boundary for.
        boundary_file: File containing a GeoDataFrame with a boundary polygon or multipolygon.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with the boundary polygon.
    """
    if local_authorities:
        boundary_gdf = load_boundaries.load_gdf_local_authority_boundaries(
            local_authorities
        )
    else:
        boundary_gdf = gpd.read_file(boundary_file)

    boundary_gdf = boundary_gdf.to_crs(27700)
    return boundary_gdf.geometry.union_all()


def clip_and_save_dataset(
    input_filepath, boundary_geometry, layer=None, save_output=False
):
    """
    Clips a geospatial input file based on the boundary geometry and saves a file to S3.

    Args:
        input_filepath (str): name of the file to clip.
        boundary_geometry (gpd.GeoDataFrame): boundary geometry to clip to.
        layer: name of the layer of the input file to load (if loading a geopackage).
        save_output (bool): to save output to S3.

    Returns:
        gpd.GeoDataframe | pd.DataFrame clipped to the boundary geometry and saved to S3.
    """
    print(f"Processing {input_filepath}...")

    input_path = Path(input_filepath)
    file_extension = input_path.suffix.lower()

    # handle different file types with their dedicated getters
    try:
        if file_extension == ".csv":
            df = base_getters.load_df_from_s3(str(input_filepath))
            is_geodata = False

        elif file_extension == ".zip":
            filename = os.path.basename(input_path).split("_csv")[0]
            df = base_getters.get_df_from_zip_csv_s3(
                str(input_filepath), extract_file=f"{filename}.csv"
            )
            is_geodata = False

        elif file_extension == ".gpkg":
            print(f"Loading layer '{layer}' from GeoPackage...")
            gdf = base_getters.get_gdf_from_gpkg_s3_path(
                str(input_filepath), layer=layer
            )
            is_geodata = True

        else:
            gdf = gpd.read_file(str(input_filepath))
            is_geodata = True

        # convert data that contains X and Y coordinates to a gdf in order to clip to boundary
        if not is_geodata:
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["X_COORDINATE"], df["Y_COORDINATE"]),
                crs=27700,
            )
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")

    clipped_gdf = gpd.clip(gdf, boundary_geometry)

    output_filename = f"test_{input_path.name}"
    output_filepath = input_path.parent / output_filename

    # drop geometry column from data containing X and Y columns for consistency with input dataset
    if not is_geodata:
        output_df = clipped_gdf.drop(columns="geometry")
    else:
        output_df = clipped_gdf

    if save_output:
        save_utils.save_to_s3(output_df, str(output_filepath))

    return output_df


def filter_epc_by_uprn(
    input_filepath, reference_df, epc_type, country=None, save_output=False
):
    """
    Filters a dataset containing UPRNs to only those UPRNs found in a reference dataset.

    Args:
        input_filepath (str): input file to filter
        reference_df (pd.DataFrame | gpd.GeoDataFrame): reference dataframe containing a list of UPRNs to filter the input file to.
        save_output (bool): to save output to S3.
    """

    print(f"Processing {input_filepath}...")

    # UPRNs we want to filter to
    valid_uprns = set(reference_df["UPRN"].dropna())
    valid_uprns_list = list(valid_uprns)

    # filter input to just these UPRNs
    # handle different EPC files
    if epc_type == "domestic":
        target_df = base_getters.load_df_from_s3(input_filepath, columns="UPRN")
        filtered_df = target_df.filter(
            pl.col("UPRN")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64)
            .is_in(valid_uprns_list)
        )
    else:
        if country == "EW":
            target_df = base_getters.load_df_from_s3(input_filepath, columns="UPRN")
            filtered_df = target_df.filter(
                pl.col("UPRN")
                .cast(pl.Float64, strict=False)
                .cast(pl.Int64)
                .is_in(valid_uprns_list)
            )
        else:
            target_df = base_getters.load_df_from_s3(
                input_filepath, columns="OSG_REFERENCE_NUMBER"
            )

            filtered_df = target_df.filter(
                pl.col("OSG_REFERENCE_NUMBER")
                .cast(pl.Float64, strict=False)
                .cast(pl.Int64)
                .is_in(valid_uprns_list)
            )

    if save_output:
        input_path = Path(input_filepath)
        output_filename = f"uprn_filtered_{input_path.name}"
        output_path = input_path.parent / output_filename

        save_utils.save_to_s3(filtered_df, str(output_path))
        return output_path

    return None


def filter_country_mapping_by_uprn(input_filepath, reference_df, save_output=False):
    """
    Filters a dataset containing UPRNs to only those UPRNs found in a reference dataset.

    Args:
        input_filepath (str): input file to filter
        reference_filepath (str): reference file containing a list of UPRNs to filter the input file to.
        save_output (bool): to save output to S3.
    """

    print(f"Processing {input_filepath}...")

    # UPRNs we want to filter to
    valid_uprns = set(reference_df["UPRN"].dropna())

    print("Loading UPRN to country mapping...")

    # country mapping file getter
    s3_client = boto3.client("s3")

    path = input_filepath
    bucket_name = path.split("s3://")[1].split("/")[0]
    prefix = path.split(f"s3://{bucket_name}/")[1]

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = [
        f"s3://{bucket_name}/{obj['Key']}"
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]

    target_df = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True,
    )

    # filter dataframe to a set of UPRNs
    filtered_df = target_df[target_df["UPRN"].isin(valid_uprns)]

    if save_output:
        input_path = Path(input_filepath)
        output_filename = f"uprn_filtered_{input_path.name}"
        output_path = input_path.parent / output_filename

        save_utils.save_to_s3(filtered_df, str(output_path))
        return output_path

    return filtered_df


def parse_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local_authorities", type=str)
    group.add_argument("--input_geometry", type=str)
    parser.add_argument("--save", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    GEOSPATIAL_FILES_TO_PROCESS = [
        {"path": config["data"]["geodata"]["uk_osopen_uprn"], "layer": None},
        {"path": config["data"]["geodata"]["UK_poi_locations"], "layer": "poi_uk"},
        {"path": config["data"]["geodata"]["gb_code_points"], "layer": "codepoint"},
        {
            "path": config["data"]["geodata"]["gb_spatial_signatures"]["full"],
            "layer": None,
        },
        {
            "path": config["data"]["geodata"]["gb_spatial_signatures"]["simplified"],
            "layer": None,
        },
        {"path": config["data"]["processed"]["poi_anchor_properties"], "layer": None},
    ]

    EPC_FILES_TO_PROCESS = [
        {
            "path": config["data"]["epc"]["domestic"],
            "epc_type": "domestic",
            "country": None,
        },
        {
            "path": config["data"]["epc"]["commercial"]["EW"],
            "epc_type": "commercial",
            "country": "EW",
        },
        {
            "path": config["data"]["epc"]["commercial"]["S"],
            "epc_type": "commercial",
            "country": "S",
        },
    ]

    COUNTRY_MAP_FILE_TO_PROCESS = config["data"]["geodata"]["gb_uprn_country_mapping"]

    reference_uprn_df = None

    boundary_gdf = get_boundary(
        local_authorities=args.local_authorities, boundary_file=args.input_geometry
    )

    for file_info in GEOSPATIAL_FILES_TO_PROCESS:
        clipped_df = clip_and_save_dataset(
            input_filepath=file_info["path"],
            boundary_geometry=boundary_gdf,
            layer=file_info["layer"],
            save_output=args.save,
        )
        if file_info["path"] == config["data"]["geodata"]["uk_osopen_uprn"]:
            print("Saving clipped OS Open UPRNs in memory")
            reference_uprn_df = clipped_df
            print(reference_uprn_df.head())

    for file_info in EPC_FILES_TO_PROCESS:
        filter_epc_by_uprn(
            input_filepath=file_info["path"],
            reference_df=reference_uprn_df,
            epc_type=file_info["epc_type"],
            country=file_info["country"],
            save_output=args.save,
        )

    filter_country_mapping_by_uprn(
        input_filepath=COUNTRY_MAP_FILE_TO_PROCESS,
        reference_df=reference_uprn_df,
        save_output=args.save,
    )
