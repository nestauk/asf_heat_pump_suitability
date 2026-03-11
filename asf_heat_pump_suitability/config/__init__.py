"""Load and expose the base YAML configuration as a subscriptable module.

Replaces this module at import time with a :class:`_ConfigModule` that supports
dict-style access (e.g. ``config["inputs"]["geodata"]["osopen_uprn"]``), while
still allowing normal submodule imports (e.g.
``from asf_heat_pump_suitability.config.settings import Settings``).
"""

import sys
import types
from pathlib import Path

import yaml


class _ConfigModule(types.ModuleType):
    def __init__(self, name: str, path, data: dict) -> None:
        super().__init__(name)
        self.__path__ = path  # required for submodule discovery (e.g. settings.py)
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self) -> str:
        return f"<ConfigModule {self.__name__!r}>"


with open(Path(__file__).parent / "base.yaml") as f:
    _data = yaml.load(f.read(), Loader=yaml.FullLoader)

sys.modules[__name__] = _ConfigModule(__name__, __path__, _data)
