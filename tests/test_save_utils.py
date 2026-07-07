"""
Tests for asf_heat_pump_suitability.utils.save_utils and the dated release
directory convention for output datasets.

Run:
pytest tests/test_save_utils.py
"""

from datetime import datetime

import pytest

from asf_heat_pump_suitability import config
from asf_heat_pump_suitability.utils import save_utils


def test_get_str_release_date_defaults_to_today():
    """None defaults to today's date in YYYYMMDD format."""
    assert save_utils.get_str_release_date(None) == datetime.today().strftime(
        save_utils.RELEASE_DATE_FORMAT
    )


def test_get_str_release_date_returns_valid_date_unchanged():
    """A valid YYYYMMDD date string is returned as-is."""
    assert save_utils.get_str_release_date("20260708") == "20260708"


@pytest.mark.parametrize(
    "invalid_release_date",
    ["2026-07-08", "8 July 2026", "202607", "20261332", ""],
)
def test_get_str_release_date_raises_for_invalid_date(invalid_release_date):
    """Strings that are not valid YYYYMMDD dates raise ValueError."""
    with pytest.raises(ValueError):
        save_utils.get_str_release_date(invalid_release_date)


def test_output_dataset_templates_format_to_dated_release_paths():
    """Every output dataset path template resolves to a {la_slug}/{YYYYMMDD}/ release directory."""
    release_date = "20260708"
    for name, template in config["output"]["dataset"].items():
        path = template.format(
            local_authority="plymouth",
            local_authorities="plymouth",
            tolerance_m=5,
            release_date=release_date,
        )
        assert f"/plymouth/{release_date}/" in path, name
