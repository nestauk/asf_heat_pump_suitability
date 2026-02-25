"""Pydantic-settings configuration model for the heat pump suitability pipeline."""

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_BASE_YAML = Path(__file__).parent / "base.yaml"


class GeoDataPaths(BaseSettings):
    """S3 paths to geodata inputs."""

    uk_osopen_uprn: str
    gb_os_openmap_local: str
    grid_square_os_openmap_local: str
    boundaries: dict
    heat_network_zones: dict
    gb_spatial_signatures: dict
    inspire_land_registry: dict = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class EpcPaths(BaseSettings):
    """S3 paths to EPC data."""

    domestic: str
    commercial: str

    model_config = {"extra": "allow"}


class ProcessedPaths(BaseSettings):
    """S3 paths to preprocessed data."""

    non_domestic_poi_categories: str
    plymouth_residential_uprns: str

    model_config = {"extra": "allow"}


class DataPaths(BaseSettings):
    """S3 paths to all input data."""

    geodata: GeoDataPaths
    epc: EpcPaths
    processed: ProcessedPaths

    model_config = {"extra": "allow"}


class GridSquares(BaseSettings):
    """OS National Grid square codes per area."""

    plymouth: str
    plymouth_similar_cities: list[str]
    sampling_areas: list[str]

    model_config = {"extra": "allow"}


class ConstantConfig(BaseSettings):
    """Fixed constants used by the pipeline."""

    grid_squares: GridSquares
    plymouth_similar_cities: list[str]
    sampling_areas: list[str]
    target_crs: int

    model_config = {"extra": "allow"}


class OutputPaths(BaseSettings):
    """S3 paths for pipeline outputs."""

    residential_uprns_template: str = (
        "s3://asf-heat-pump-suitability/local_heat_planning/outputs/{area}_residential_uprns.parquet"
    )
    features_template: str = (
        "s3://asf-heat-pump-suitability/local_heat_planning/outputs/{uprns_stem}_with_features.parquet"
    )
    block_of_flats_model: str = (
        "s3://asf-heat-pump-suitability/local_heat_planning/outputs/models/block_of_flats_building_classifier.pkl"
    )

    model_config = {"extra": "allow"}


class Settings(BaseSettings):
    """Top-level pipeline configuration.

    Loaded from config/base.yaml; any field can be overridden by an
    environment variable of the same name (case-insensitive).
    """

    data_mode: str = Field(default="s3", description="'s3' or 'local'")
    s3_bucket: str = Field(default="asf-heat-pump-suitability")
    aws_endpoint_url: str | None = Field(default=None)
    orbit_env: str = Field(default="prod")

    data: DataPaths
    constant: ConstantConfig
    output: OutputPaths = OutputPaths()
    data_source: dict = Field(default_factory=dict)
    mapping: dict = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow", "env_prefix": ""}

    @classmethod
    def from_yaml(cls, path: Path = _BASE_YAML) -> "Settings":
        """Load settings from YAML file, allowing env var overrides."""
        raw = _load_yaml(path)
        return cls(**raw)


def load_settings(path: Path = _BASE_YAML) -> Settings:
    """Load and return the pipeline settings singleton."""
    return Settings.from_yaml(path)
