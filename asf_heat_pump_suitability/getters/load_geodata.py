import logging
from pathlib import Path

import polars as pl

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.getters import base_getters

logger = logging.getLogger(__name__)


def load_df_osopen_uprn(**kwargs) -> pl.DataFrame:
    """Get raw OS (Ordnance Survey) Open UPRN dataset containing latitude and longitude and
    British National Grid X and Y coordinates for all UPRNs in Great Britain.

    Args:
        **kwargs for pl.read_csv

    Returns:
        pl.DataFrame: raw OS Open UPRN dataset with lat/lon and x/y coordinates for every UPRN
    """
    print("Loading OSOpen UPRNs...")
    path = config["inputs"]["geodata"]["osopen_uprn"]
    filename = Path(path).name.split("_csv")[0]
    df = base_getters.get_df_from_zip_csv_s3(
        path,
        extract_file=f"{filename}.csv",
        **kwargs,
    )
    return df
