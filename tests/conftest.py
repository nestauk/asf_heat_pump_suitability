"""Shared pytest fixtures for asf_heat_pump_suitability tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the committed fixtures directory.

    Returns:
        Path: Path to tests/fixtures/.
    """
    return FIXTURES_DIR
