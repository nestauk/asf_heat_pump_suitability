"""
Creates a manifest.json file with metadata about all local authority datasets.

manifest.json is a list of dictionaries, one for each local authority, where each dictionary contains the following keys:
- "id": id of the local authority (e.g. "plymouth" or "vale_of_glamorgan")
- "local_authority": local authority name (e.g. "Plymouth" or "Vale of Glamorgan")
- "geojson_url": url to the geojson file in the front-end S3 bucket

This script is run after all local authority datasets have been created and saved to the front-end S3 bucket (with compute_contextual_features.py).

To run this script, ensure you have the following environment variables set in your .env file:
- front_end_s3_bucket: name of the S3 bucket where the manifest.json file will be saved
- front_end_staging_s3_path: path where the manifest.json file will be saved

You can then run this script with `python asf_heat_pump_suitability/pipeline/run/create_manifest.py`
"""

if __name__ == "__main__":
    from dotenv import load_dotenv
    import boto3
    import os
    from asf_heat_pump_suitability.utils import save_utils, s3_utils

    # Load environment variables from .env file
    load_dotenv()

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Load file names in s3 path
    front_end_s3_bucket = os.environ.get("front_end_s3_bucket")
    front_end_staging_s3_path = os.environ.get("front_end_staging_s3_path")

    geojson_file_paths_list = s3_utils.fetch_list_file_paths_from_s3_folder(
        s3_client=s3_client,
        s3_bucket=front_end_s3_bucket,
        path_folder=front_end_staging_s3_path,
        file_type=".geojson",
    )

    geojson_s3_url_prefix = f"https://{front_end_s3_bucket}.s3.eu-west-2.amazonaws.com/{front_end_staging_s3_path}"

    geojson_dict_list = []
    for file_path in geojson_file_paths_list:
        file_name = file_path.split("/")[-1]
        local_authority = file_name.split("_clusters")[0]

        # if file name does not follow expected format, raise error
        if len(local_authority) == len(file_name):
            raise ValueError(
                f"File name {file_name} does not follow expected format. Unable to extract local authority name."
            )

        s3_url = os.path.join(
            geojson_s3_url_prefix,
            file_name,
        )

        geojson_dict_list.append(
            {
                "id": local_authority,
                "local_authority": local_authority.replace("_", " ").title(),
                "geojson_url": s3_url,
            }
        )

    manifest_path = os.path.join(
        "s3://", front_end_s3_bucket, front_end_staging_s3_path, "manifest.json"
    )

    # Save the list of dictionaries as a manifest.json file in the front-end S3 bucket
    save_utils.save_to_s3(
        geojson_dict_list,
        manifest_path,
    )

    print(
        f"Manifest file created and saved to {manifest_path} in S3 bucket {front_end_s3_bucket}."
    )
