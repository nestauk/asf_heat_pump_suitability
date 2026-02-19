"""asf_heat_pump_suitability."""

from pathlib import Path

import yaml

__version__ = "0.1.0"


def _load_yaml(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return None


# Raw YAML config dict — used by getters and pipeline utilities.
# For new code, prefer asf_heat_pump_suitability.config.settings.load_settings().
config: dict = _load_yaml(Path(__file__).parent / "config/base.yaml") or {}
