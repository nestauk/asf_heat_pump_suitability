"""CLI: Stream INSPIRE land registry files from government websites to S3.

Streams INSPIRE land parcel polygon files to the ``asf-heat-pump-suitability``
S3 bucket. Files are unzipped during streaming.

- **Scotland**: Partitioned by Registration County (shapefile packages) from the
  Registers of Scotland (ROS) website.
- **England and Wales**: Partitioned by Local Authority (GML format) from the
  HMLR Use Land and Property Data service.

**When to run**: Run this script when the INSPIRE land parcel data needs to be
refreshed (e.g. quarterly updates). It is a one-off setup step and does not need
to run as part of the regular pipeline.

Example usage:

    # Stream both Scotland and England & Wales
    uv run python pipeline/setup/stream_inspire_files.py --nations all

    # Stream England & Wales only
    uv run python pipeline/setup/stream_inspire_files.py --nations ew

    # Stream Scotland only
    uv run python pipeline/setup/stream_inspire_files.py --nations s
"""

import time
import zipfile
from typing import Annotated

import boto3
import regex as re
import requests
import typer
from bs4 import BeautifulSoup
from tqdm import tqdm

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters

app = typer.Typer(help=__doc__)

NationsChoice = Annotated[
    str,
    typer.Option(
        "--nations",
        "-n",
        help="Nations to download INSPIRE land registry files for.",
        click_type=typer.Choice(["ew", "s", "all"]),
    ),
]


@app.command()
def main(
    nations: NationsChoice = "all",
) -> None:
    """Stream INSPIRE land registry files to S3."""
    s3 = boto3.client("s3")
    bucket = "asf-heat-pump-suitability"

    if nations in ["s", "all"]:
        _stream_scotland(s3=s3, bucket=bucket)

    if nations in ["ew", "all"]:
        _stream_england_wales(s3=s3, bucket=bucket)

    print("Done.")


def _stream_scotland(s3: boto3.client, bucket: str) -> None:
    """Stream Scottish INSPIRE files from Registers of Scotland to S3.

    Args:
        s3: Boto3 S3 client.
        bucket: Name of the S3 bucket to upload to.
    """
    ros_url = config["data_source"]["S_ros_inspire_url"]
    print(f"Fetching Scottish INSPIRE file list from: {ros_url}")

    page = requests.get(ros_url)
    soup = BeautifulSoup(page.content, "html.parser")

    pattern = re.compile(r'https:.+?(?=")')
    url_prefix = pattern.search(soup.find("script", string=pattern).text).group(0)
    ids = {v.contents[0]: v["value"] for v in soup.find_all("option") if v["value"]}
    download_urls = {area: url_prefix + url for area, url in ids.items()}

    print(f"Streaming {len(download_urls)} Scottish registration counties to S3...")
    for area, url in tqdm(download_urls.items()):
        content = base_getters.get_content_from_url(url)
        z = zipfile.ZipFile(content)
        for file in z.namelist():
            with z.open(file) as f:
                s3.upload_fileobj(
                    Fileobj=f,
                    Bucket=bucket,
                    Key=f"source_data/inspire_scotland/{area}/{file}",
                )


def _stream_england_wales(s3: boto3.client, bucket: str) -> None:
    """Stream England and Wales INSPIRE files from HMLR to S3.

    Args:
        s3: Boto3 S3 client.
        bucket: Name of the S3 bucket to upload to.
    """
    ew_url = config["data_source"]["EW_inspire_url"]
    print(f"Fetching England & Wales INSPIRE file list from: {ew_url}")

    page = requests.get(url=ew_url, headers={"User-Agent": "Data collection for research."})
    soup = BeautifulSoup(markup=page.content, features="html.parser")

    urls = [a.get("href") for a in soup.find_all(name="a", attrs={"aria-current": False})]
    urls = [url for url in urls if url and url.endswith(".zip")]
    url_prefix = "https://use-land-property-data.service.gov.uk/"

    print(f"Streaming {len(urls)} England & Wales local authority files to S3...")
    for url in tqdm(urls):
        area = url.split("/")[-1].split(".zip")[0]
        content = base_getters.get_content_from_url(url_prefix + url)
        z = zipfile.ZipFile(content)
        gml_files = [f for f in z.namelist() if f.endswith(".gml")]
        for file in gml_files:
            with z.open(file) as f:
                s3.upload_fileobj(
                    Fileobj=f,
                    Bucket=bucket,
                    Key=f"source_data/inspire_ew/{area}.gml",
                )
        time.sleep(1)


if __name__ == "__main__":
    app()
