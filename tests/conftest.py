"""Shared pytest fixtures for the heat pump suitability test suite.

Session-scoped moto fixture:
- Starts a mock S3 context
- Creates the asf-heat-pump-suitability bucket
- Uploads committed fixture files to their expected S3 paths
- Yields (all tests share the mock)
- Tears down the mock

All pipeline code runs against this mock transparently via AWS_ENDPOINT_URL.
"""

import os
from pathlib import Path

import boto3
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BUCKET = "asf-heat-pump-suitability"


@pytest.fixture(scope="session", autouse=True)
def aws_credentials() -> None:
    """Set dummy AWS credentials for moto."""
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-west-2")


@pytest.fixture(scope="session")
def s3_bucket(aws_credentials: None) -> None:  # noqa: ANN001
    """Session-scoped moto S3 mock with fixture data loaded into the bucket."""
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-2")
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )

        # Upload all committed fixture files to their expected S3 paths
        for fixture_file in FIXTURES_DIR.rglob("*"):
            if fixture_file.is_file() and not fixture_file.name.startswith("."):
                relative = fixture_file.relative_to(FIXTURES_DIR)
                s3_key = str(relative)
                s3.upload_file(str(fixture_file), BUCKET, s3_key)

        yield s3
