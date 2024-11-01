"""
Script to stream INSPIRE files for Scotland from ROS webpage and/or INSPIRE files for England and Wales
from government website to S3 asf-heat-pump-suitability bucket.

Scottish INSPIRE files are partitioned according to Registration Counties in shapefile packages.
England and Wales INSPIRE files are partitioned according to Local Authorities in gml format.

python -i asf_heat_pump_suitability/pipeline/run_scripts/run_download_inspire.py -n all

[Set -n nation flag to "ew" or "s" for streaming either England and Wales or Scotland INSPIRE files only.]
"""

from bs4 import BeautifulSoup
import requests
import regex as re
import boto3
import argparse
import time
from tqdm import tqdm
import zipfile
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


def parse_arguments() -> argparse.Namespace:
    """
    Create ArgumentParser and parse.

    Returns:
        argparse.Namespace: populated `Namespace`
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--nations",
        help="Nations to download INSPIRE land registry files for out of England and Wales; Scotland; or all.",
        type=str,
        choices=["ew", "s", "all"],
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    s3 = boto3.client("s3")
    bucket = "asf-heat-pump-suitability"

    if args.nations in ["s", "all"]:
        ros_url = config["data_source"]["S_ros_inspire_url"]

        page = requests.get(ros_url)
        soup = BeautifulSoup(page.content, "html.parser")

        # Generate download URLs for each Scottish registration county
        pattern = re.compile(f'https:.+?(?=")')
        url_prefix = pattern.search(soup.find("script", string=pattern).text).group(0)
        ids = {v.contents[0]: v["value"] for v in soup.find_all("option") if v["value"]}
        download_urls = {area: url_prefix + url for area, url in ids.items()}

        # Unzip shapefile at URL and stream to S3
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

    if args.nations in ["ew", "all"]:
        page = requests.get(
            url=config["data_source"]["EW_inspire_url"],
            headers={"User-Agent": "Data collection for research."},
        )
        soup = BeautifulSoup(markup=page.content, features="html.parser")

        # Generate download URLs for all available England and Wales local authorities
        urls = []
        for a in soup.find_all(name="a", attrs={"aria-current": False}):
            urls.append(a.get("href"))
        urls = [url for url in urls if url.endswith(".zip")]
        url_prefix = "https://use-land-property-data.service.gov.uk/"

        # Unzip gml file at URL and stream to S3
        for url in tqdm(urls):
            area = url.split("/")[-1].split(".zip")[0]
            content = base_getters.get_content_from_url(url_prefix + url)
            z = zipfile.ZipFile(content)
            file = [file for file in z.namelist() if file.endswith(".gml")][0]
            with z.open(file) as f:
                s3.upload_fileobj(
                    Fileobj=z.open(file),
                    Bucket=bucket,
                    Key=f"source_data/inspire_ew/{area}.gml",
                )
            time.sleep(1)
