"""
Aggregate the build year data up to local authority level.
"""

import pandas as pd

if __name__ == "__main__":

    lad_to_lsoa = pd.read_csv(
        "s3://asf-heat-pump-suitability/source_data/2021_vApr2023_ons_lsoa_to_lad_lookup_EW.csv"
    )
    lsoa_build_year = pd.read_csv(
        "s3://asf-heat-pump-suitability/source_data/2015cdrc_dwelling_ages_E_W.csv"
    )

    lsoa_to_lad_dict = dict(zip(lad_to_lsoa["LSOA21CD"], lad_to_lsoa["LAD23CD"]))
    lad_name_dict_dict = dict(zip(lad_to_lsoa["LAD23CD"], lad_to_lsoa["LAD23NM"]))

    lsoa_build_year["LAD23CD"] = lsoa_build_year["AREA_CODE"].map(lsoa_to_lad_dict)
    lsoa_build_year["LAD23NM"] = lsoa_build_year["LAD23CD"].map(lad_name_dict_dict)

    columns = [
        "BP_PRE_1900",
        "BP_1900_1918",
        "BP_1919_1929",
        "BP_1930_1939",
        "BP_1945_1954",
        "BP_1955_1964",
        "BP_1965_1972",
        "BP_1973_1982",
        "BP_1983_1992",
        "BP_1993_1999",
        "BP_2000_2009",
        "BP_2010_2015",
        "BP_UNKNOWN",
        "ALL_PROPERTIES",
    ]

    la_build_year = (
        lsoa_build_year.groupby(["LAD23CD", "LAD23NM"])[columns].sum().reset_index()
    )
    la_build_year.to_csv(
        "s3://asf-heat-pump-suitability/source_data_minor_edits/2015cdrc_dwelling_ages_E_W_per_la.csv"
    )
