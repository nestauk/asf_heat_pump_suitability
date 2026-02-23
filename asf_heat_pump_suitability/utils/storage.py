"""Storage utilities for the pipeline.

Two complementary mechanisms handle local development and testing:

1. ``get_path()`` — path-string routing
   Converts an S3 URI into a local ``outputs/`` path when ``DATA_MODE=local``.
   ``save_utils.save_to_s3()`` detects the local path and writes directly via
   Polars, bypassing S3 entirely.  This is the primary mechanism for **output
   writes** during local development.

   Limitation: input paths (data sources read by the pipeline) are still
   hardcoded S3 URIs and are not yet routed through ``get_path()``, so the
   pipeline cannot run end-to-end in local mode without real or mocked S3 data.

2. ``mock_aws_if_local()`` — botocore-level interception
   Activates moto when ``DATA_MODE != 's3'``, patching botocore globally so
   all boto3/s3fs calls hit an in-memory fake S3 rather than real AWS.  This
   acts as a safety net against accidental writes to the real bucket during
   local development.

   In the **test suite** (``tests/conftest.py``), ``mock_aws()`` is used
   directly (not via this helper) and fixture files are pre-loaded into the
   mock bucket, so pipeline reads also succeed.  ``mock_aws_if_local()`` is
   never invoked during tests because ``DATA_MODE`` is not set to ``local``
   there.

   For local development runs, ``mock_aws_if_local()`` intercepts any stray
   S3 calls, but reads of input data will fail with ``NoSuchKey`` unless the
   moto bucket has been pre-populated separately.

All S3 clients and filesystems must be obtained through ``get_s3fs()`` and
``get_boto3_client()`` so that both mechanisms work correctly.
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

    When ``DATA_MODE=local``, patches botocore globally so all boto3/s3fs
    calls hit an in-memory fake S3 rather than real AWS.  This prevents
    accidental writes to the real S3 bucket during local development.

    When ``DATA_MODE=s3`` (the default for cloud runs), this is a no-op.

    Note: this is **not** used by the test suite.  Tests activate moto
    directly via the ``s3_bucket`` fixture in ``tests/conftest.py``, which
    also pre-populates the mock bucket with fixture data so that pipeline
    reads succeed.  ``mock_aws_if_local()`` leaves the moto bucket empty, so
    pipeline reads of input data will still fail in local mode unless the
    bucket is pre-populated separately.

    Example::

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
    """Return the appropriate path for an S3 URI.

    When ``DATA_MODE=local``, strips the ``s3://asf-heat-pump-suitability/``
    prefix and returns a path under ``outputs/`` on the local filesystem,
    creating parent directories as needed.  ``save_utils.save_to_s3()``
    detects the local path and writes via Polars native I/O, so no S3
    connection is required for output writes.

    When ``DATA_MODE=s3`` (default), returns ``key`` unchanged.

    Args:
        key: S3 URI (``s3://asf-heat-pump-suitability/...``).
        config: Loaded Settings instance.

    Returns:
        str: S3 URI (cloud mode) or local ``outputs/...`` path (local mode).
    """
    if getattr(config, "data_mode", "s3") == "local":
        local_key = key.replace("s3://asf-heat-pump-suitability/", "")
        local_path = Path("outputs") / local_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        return str(local_path)
    return key
