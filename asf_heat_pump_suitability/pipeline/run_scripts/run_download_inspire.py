"""
Script to download INSPIRE files for Scotland from ROS webpage and save to S3 asf-heat-pump-suitability bucket.
"""

from bs4 import BeautifulSoup
import requests
import regex as re
import boto3
from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters


if __name__ == "__main__":
    ros_url = config["data_source"]["S_ros_inspire_url"]
    bucket = "asf-heat-pump-suitability"

    page = requests.get(ros_url)
    soup = BeautifulSoup(page.content, "html.parser")

    pattern = re.compile(f'https:.+?(?=")')
    url = pattern.search(soup.find("script", string=pattern).text).group(0)
    ids = {v.contents[0]: v["value"] for v in soup.find_all("option") if v["value"]}
    download_urls = {k: url + v for k, v in ids.items()}

    s3 = boto3.client("s3")
    for area, url in download_urls.items():
        content = base_getters.get_content_from_url(url)
        s3.upload_fileobj(
            content, bucket, f"source_data/inspire_zips_scotland/{area}.zip"
        )
