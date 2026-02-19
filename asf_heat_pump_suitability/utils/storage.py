"""Thin path-only storage abstraction for the pipeline.

All pipeline S3 I/O obtains paths through this module so that switching
between real S3 and a local filesystem (DATA_MODE=local) requires no changes
in pipeline code.
"""

from pathlib import Path


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
