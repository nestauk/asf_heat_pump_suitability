"""Storage utilities for the pipeline.

All pipeline S3 I/O should obtain s3fs filesystems and boto3 clients through
this module. This ensures that the AWS_ENDPOINT_URL environment variable is
respected, which is required for moto-based testing and localstack-based local
development.

All pipeline S3 I/O obtains paths through this module so that switching
between real S3 and a local filesystem (DATA_MODE=local) requires no changes
in pipeline code.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import boto3
import s3fs


def get_s3fs() -> s3fs.S3FileSystem:
    """Create an S3FileSystem, routing to AWS_ENDPOINT_URL when set.

    When AWS_ENDPOINT_URL is set (e.g. for moto or localstack), all S3
    operations are directed to that endpoint rather than real AWS. This is
    required for credential-free CI tests.

    Returns:
        s3fs.S3FileSystem: Configured filesystem instance.
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    if endpoint_url:
        return s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint_url})
    return s3fs.S3FileSystem()


def get_boto3_client(service: str) -> boto3.client:
    """Create a boto3 client, routing to AWS_ENDPOINT_URL when set.

    Args:
        service: AWS service name (e.g. "s3").

    Returns:
        boto3.client: Configured client instance.
    """
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
    return boto3.client(service, endpoint_url=endpoint_url)


@contextmanager
def mock_aws_if_local() -> Generator[None, None, None]:
    """Activate a moto S3 mock when DATA_MODE is not 's3'.

    Pipeline entry scripts should wrap their ``run()`` call with this context
    manager. When ``DATA_MODE=local`` (or any non-s3 value), all boto3 and
    s3fs calls are intercepted by moto so no real AWS credentials are required.
    When ``DATA_MODE=s3`` (the default for cloud runs), this is a no-op.

    Example usage in a pipeline entry script::

        if __name__ == "__main__":
            with mock_aws_if_local():
                run(...)

    Yields:
        None
    """
    if os.environ.get("DATA_MODE", "s3") != "s3":
        from moto import mock_aws

        with mock_aws():
            yield
    else:
        yield


def get_path(key: str, config: "Settings") -> str:  # noqa: F821
    """Return the appropriate path for a data key.

    When DATA_MODE=local, returns a local filesystem path under outputs/.
    When DATA_MODE=s3 (default), returns the configured S3 URI.

    Args:
        key: Dot-separated key into the config output paths, or a raw S3 URI.
        config: Loaded Settings instance.

    Returns:
        str: Path string suitable for use with polars, geopandas, or s3fs.
    """
    if getattr(config, "data_mode", "s3") == "local":
        local_key = key.replace("s3://asf-heat-pump-suitability/", "")
        local_path = Path("outputs") / local_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return str(local_path)
    return key
