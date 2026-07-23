"""
Stream per-grid-square OS OpenData products from the OS Downloads API to S3.

For each requested product the script queries the public OS Downloads API (no
auth required for OpenData products), selects the ESRI Shapefile downloads and
streams every zip member (shapefile plus all sidecars, licence and readme) to
the product's version-named S3 prefix. The dated segment of each prefix is the
API's product `version` verbatim (e.g. "2026-04") so the prefix names the OS
release, not the download event. Each product keeps its current on-S3 layout:

- OpenMapLocal:   opmplc_essh_gb/{version}/data/{square}/{square}_{layer}.*
- OpenRoads:      oproad_essh_gb/{version}/data/{square}_{layer}.* (fanned out
                  from the single GB zip; the API offers no per-square roads)
- OpenGreenspace: opgrsp_essh_gb/{version}/{square}/data/{square}_{layer}.*

Downloaded zips are md5-checked against the API listing. After uploading, the
script lists S3 under each new prefix and reconciles it against the keys
derived from the API's offered areas (for roads: the GB zip's members),
exiting non-zero on any mismatch.

By default the script performs a dry run: it lists each product's release
version, tiles and sizes without touching S3.

Run:
python asf_heat_pump_suitability/pipeline/run/stream_os_open_data.py --products OpenMapLocal OpenRoads OpenGreenspace

To upload to S3, add the --save flag.

Set --destination_root to override the configured S3 destination root (e.g.
a test prefix such as
"s3://asf-local-heat-planning-tool/inputs/geodata/test_419_os_downloads/").
"""

import argparse
import hashlib
import io
import logging
import sys
import zipfile
from typing import Iterable, Optional

import boto3
import requests

from asf_heat_pump_suitability import config

SHAPEFILE_FORMAT = "ESRI® Shapefile"
GB_AREA = "GB"
# Products the API offers only as a single GB zip whose members are per-square
# files; these are fanned out rather than downloaded per area
GB_ZIP_PRODUCTS = {"OpenRoads"}


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()
    product_choices = list(config["os_downloads"]["products"])

    parser.add_argument(
        "--products",
        help="OS Downloads API product IDs to stream to S3. Defaults to all configured products.",
        type=str,
        nargs="+",
        choices=product_choices,
        default=product_choices,
    )

    parser.add_argument(
        "--save",
        help="If --save is set, upload to S3. Otherwise perform a dry run: list tiles, sizes and release version only.",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--destination_root",
        help="Override the configured S3 destination root, e.g. a test prefix for a rehearsal run.",
        required=False,
        type=str,
        default=None,
    )

    return parser.parse_args()


def get_dict_product_details(api_url: str, product: str) -> dict:
    """
    Get product details from the OS Downloads API.

    Args:
        api_url: OS Downloads API base URL
        product: OS Downloads API product ID, e.g. "OpenMapLocal"

    Returns:
        dict: product details including `version` and offered `areas`
    """
    response = requests.get(f"{api_url}/products/{product}", timeout=60)
    response.raise_for_status()
    return response.json()


def get_list_product_downloads(api_url: str, product: str) -> list:
    """
    Get the list of downloads offered for a product from the OS Downloads API.

    Args:
        api_url: OS Downloads API base URL
        product: OS Downloads API product ID, e.g. "OpenMapLocal"

    Returns:
        list: download entries, each a dict with `area`, `format`, `fileName`,
            `url`, `size` and `md5`
    """
    response = requests.get(f"{api_url}/products/{product}/downloads", timeout=60)
    response.raise_for_status()
    return response.json()


def filter_list_shapefile_downloads(downloads: list, product: str) -> list:
    """
    Select the ESRI Shapefile downloads to stream for a product.

    Products in GB_ZIP_PRODUCTS keep only the single GB zip; all other
    products keep the per-area zips and drop the GB zip.

    Args:
        downloads: download entries from the OS Downloads API
        product: OS Downloads API product ID

    Returns:
        list: filtered download entries
    """
    shapefiles = [entry for entry in downloads if entry["format"] == SHAPEFILE_FORMAT]
    if product in GB_ZIP_PRODUCTS:
        return [entry for entry in shapefiles if entry["area"] == GB_AREA]
    return [entry for entry in shapefiles if entry["area"] != GB_AREA]


def generate_key_zip_member(product: str, area: str, member: str) -> Optional[str]:
    """
    Map a zip member name to its S3 key relative to the product's dated prefix.

    Reproduces each product's current on-S3 layout (see module docstring).
    Per-area zips wrap their contents in a folder such as
    "OS OpenMap Local (ESRI Shape File) HP/", which is stripped; the OpenRoads
    GB zip names some members with a leading slash, which is also stripped.

    Args:
        product: OS Downloads API product ID
        area: area code of the download the member belongs to, e.g. "HP"
        member: zip member name

    Returns:
        Optional[str]: S3 key relative to the product's dated prefix, or None
            for directory entries
    """
    member = member.lstrip("/")
    if not member or member.endswith("/"):
        return None
    if product == "OpenRoads":
        return member
    remainder = member.split("/", 1)[1] if "/" in member else member
    if not remainder:
        return None
    if product == "OpenGreenspace":
        return f"{area}/{remainder}"
    if product == "OpenMapLocal":
        if remainder.startswith("data/"):
            return f"data/{area}/{remainder.removeprefix('data/')}"
        return remainder
    raise ValueError(f"No S3 layout mapping defined for product '{product}'.")


def generate_dict_reconciliation(
    expected_keys: Iterable, actual_keys: Iterable
) -> dict:
    """
    Diff expected against actual S3 keys.

    Args:
        expected_keys: keys that should exist after the upload
        actual_keys: keys found on S3

    Returns:
        dict: {"missing": sorted keys expected but absent,
            "unexpected": sorted keys present but not expected}
    """
    expected, actual = set(expected_keys), set(actual_keys)
    return {
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }


def get_tuple_s3_bucket_prefix(s3_uri: str) -> tuple:
    """
    Split an S3 URI into bucket name and key prefix.

    Args:
        s3_uri: URI of the form "s3://bucket/key/prefix/"

    Returns:
        tuple: (bucket, key prefix)
    """
    bucket, _, prefix = s3_uri.removeprefix("s3://").partition("/")
    return bucket, prefix


def get_list_s3_keys(s3_client, bucket: str, prefix: str) -> list:
    """
    List all object keys under an S3 prefix.

    Args:
        s3_client: boto3 S3 client
        bucket: S3 bucket name
        prefix: key prefix to list under

    Returns:
        list: object keys
    """
    keys = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def verify_zip_md5(content: bytes, expected_md5: str, file_name: str) -> None:
    """
    Verify downloaded zip content against the md5 from the API listing.

    Args:
        content: downloaded zip bytes
        expected_md5: md5 hex digest from the API download entry
        file_name: zip file name, for the error message

    Raises:
        ValueError: if the digests do not match
    """
    actual_md5 = hashlib.md5(content).hexdigest()
    if actual_md5 != expected_md5:
        raise ValueError(
            f"MD5 mismatch for {file_name}: API listing says {expected_md5}, "
            f"downloaded content is {actual_md5}."
        )


def stream_zip_members_to_s3(
    s3_client, download: dict, product: str, bucket: str, prefix: str
) -> set:
    """
    Download one zip, verify its md5 and upload its members to S3.

    Args:
        s3_client: boto3 S3 client
        download: OS Downloads API download entry
        product: OS Downloads API product ID
        bucket: destination S3 bucket
        prefix: destination key prefix (the product's dated prefix)

    Returns:
        set: S3 keys uploaded
    """
    logging.info(
        f"Downloading {download['fileName']} ({download['size'] / 1e6:.1f} MB)"
    )
    response = requests.get(download["url"], timeout=3600)
    response.raise_for_status()
    verify_zip_md5(response.content, download["md5"], download["fileName"])
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    uploaded = set()
    for member in zip_file.namelist():
        relative_key = generate_key_zip_member(product, download["area"], member)
        if relative_key is None:
            continue
        key = f"{prefix}{relative_key}"
        with zip_file.open(member) as file_obj:
            s3_client.upload_fileobj(Fileobj=file_obj, Bucket=bucket, Key=key)
        uploaded.add(key)
    return uploaded


def main() -> None:
    """Run the download/upload flow for each requested product."""
    args = parse_arguments()
    api_url = config["os_downloads"]["api_url"]
    destination_root = (
        args.destination_root or config["os_downloads"]["s3_destination_root"]
    )
    if not destination_root.endswith("/"):
        destination_root += "/"
    s3_client = boto3.client("s3") if args.save else None
    failed_products = []

    for product in args.products:
        details = get_dict_product_details(api_url, product)
        version = details["version"]
        downloads = get_list_product_downloads(api_url, product)
        selected = filter_list_shapefile_downloads(downloads, product)
        prefix_template = config["os_downloads"]["products"][product]
        destination = destination_root + prefix_template.format(version=version)

        # The downloads listing must cover exactly the areas the product offers
        offered_areas = (
            {GB_AREA}
            if product in GB_ZIP_PRODUCTS
            else set(details["areas"]) - {GB_AREA}
        )
        area_diff = generate_dict_reconciliation(
            offered_areas, {entry["area"] for entry in selected}
        )
        if area_diff["missing"] or area_diff["unexpected"]:
            logging.error(
                f"{product}: shapefile downloads do not match the product's "
                f"offered areas: {area_diff}"
            )
            sys.exit(1)

        total_mb = sum(entry["size"] for entry in selected) / 1e6
        logging.info(
            f"{product} version {version}: {len(selected)} zip(s), "
            f"{total_mb:.1f} MB -> {destination}"
        )
        for entry in selected:
            logging.info(
                f"  {entry['area']}: {entry['fileName']} "
                f"({entry['size'] / 1e6:.1f} MB)"
            )
        if not args.save:
            continue

        bucket, prefix = get_tuple_s3_bucket_prefix(destination)
        expected_keys = set()
        for entry in selected:
            expected_keys |= stream_zip_members_to_s3(
                s3_client, entry, product, bucket, prefix
            )
        actual_keys = get_list_s3_keys(s3_client, bucket, prefix)
        diff = generate_dict_reconciliation(expected_keys, actual_keys)
        if diff["missing"] or diff["unexpected"]:
            logging.error(
                f"{product}: S3 under s3://{bucket}/{prefix} does not match "
                f"the files offered by the OS Downloads API. "
                f"Missing: {diff['missing']} Unexpected: {diff['unexpected']}"
            )
            failed_products.append(product)
        else:
            logging.info(
                f"{product}: reconciliation OK, {len(actual_keys)} files at "
                f"s3://{bucket}/{prefix}"
            )

    if failed_products:
        sys.exit(1)
    if not args.save:
        logging.info("Dry run complete. Re-run with --save to upload to S3.")


if __name__ == "__main__":
    main()
