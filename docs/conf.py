"""Sphinx configuration."""
from datetime import datetime


project = "heat_pump_suitability"
author = "Nesta"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    # "sphinx_click",
    "sphinx_rtd_theme",
]
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
