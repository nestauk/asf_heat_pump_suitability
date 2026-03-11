"""Pydantic settings for asf_heat_pump_suitability pipeline.

Reads LOCAL_DEV and OUTPUT_DIR environment variables and resolves output paths.
By default LOCAL_DEV=true so pipeline outputs go to the local filesystem,
preventing accidental writes to S3 from a developer laptop.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline settings loaded from environment variables.

    Attributes:
        local_dev: When True (default), pipeline outputs are written to the local
            filesystem. Set to False for cloud/production runs that write to S3.
        output_dir: Override the base directory for resolved output paths. If not set,
            defaults to ``./outputs/`` when ``local_dev=True`` and to
            ``s3://asf-local-heat-planning-tool/outputs/`` when ``local_dev=False``.
    """

    local_dev: bool = True
    output_dir: str | None = None

    model_config = {"env_prefix": ""}

    @property
    def resolved_output_dir(self) -> str:
        """Return the resolved base output directory.

        Returns:
            str: Base directory path (local or S3) for pipeline outputs.
        """
        if self.output_dir:
            return self.output_dir
        return "./outputs/" if self.local_dev else "s3://asf-local-heat-planning-tool/outputs/"

    def resolve_output_path(self, filename: str) -> str:
        """Resolve a pipeline output filename to its full path.

        Args:
            filename: The bare filename (e.g. ``domestic_uprns.parquet``).

        Returns:
            str: Full path including the resolved base output directory.
        """
        base = self.resolved_output_dir.rstrip("/")
        return f"{base}/{filename}"
