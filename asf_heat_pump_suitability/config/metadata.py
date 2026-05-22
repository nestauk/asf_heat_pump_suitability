"""
Metadata generated for each local authority dataset with the assigned low-carbon heating technology for clusters of proprerties and contextual information about properties within the clusters and surrounding neighbourhood.

This metadata is included in the output dataset and provides information on the license, data sources, variable descriptions, and other relevant information about the dataset. The variable descriptions provide detailed explanations of each variable included in the dataset, which can help users understand the data and its potential applications.
"""

from datetime import datetime
from asf_heat_pump_suitability import config

ANCHOR_LOAD_RADIUS = config["constant"]["clustering"]["anchor_load_radius"]
COASTLINE_DISTANCE_THRESHOLD_M = config["constant"]["clustering"][
    "distance_from_coastline_threshold_m"
]
metadata = {
    "License": "This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.",
    "Date of creation": datetime.now().strftime("%Y-%m-%d"),
    "Data sources and attributions": "https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/config#readme",
    "Variable descriptions": {
        "cluster_id": "Unique identifier for each cluster.",
        "assigned_tech": "Assigned low-carbon heating technology category. Categories are: `Individual solution`, `Networked heat pump`, `Communal solution`, `District heat network`. Can additionaly contain `DESNZ_HNZ` values (if there the local authority was part of the DESNZ heat network zoning).",
        "geometry": "Polygon geometry of the cluster",
        "n_UPRNs": "Number of domestic UPRNs (properties) within the cluster",
        "n_uprns_in_listed_building": "Number of UPRNs (properties) within the cluster that are in listed buildings",
        "n_uprns_solar_pv": "Number of UPRNs (properties) within the cluster that have solar PV",
        "n_uprns_off_gas": "Number of UPRNs (properties) within the cluster that are off-gas",
        "perc_uprns_solar_pv": "Percentage of UPRNs (properties) within the cluster that have solar PV",
        "perc_uprns_off_gas": "Percentage of UPRNs (properties) within the cluster that are off-gas",
        "attachment_detached": "Number of UPRNs (properties) within the cluster that are detached",
        "attachment_end_terrace": "Number of UPRNs (properties) within the cluster that are end terrace",
        "attachment_flat": "Number of UPRNs (properties) within the cluster that are flats",
        "attachment_mid_terrace": "Number of UPRNs (properties) within the cluster that are mid terrace",
        "attachment_semi_detached": "Number of UPRNs (properties) within the cluster that are semi detached",
        "attachment_null": "Number of UPRNs (properties) within the cluster with unknown attachment type",
        "tenure_owner_occupied": "Number of UPRNs (properties) within the cluster that are owner occupied",
        "tenure_rental_(private)": "Number of UPRNs (properties) within the cluster that are private rental",
        "tenure_rental_(social)": "Number of UPRNs (properties) within the cluster that are social rental",
        "tenure_null": "Number of UPRNs (properties) within the cluster with unknown tenure",
        "perc_tenure_owner_occupied": "Percentage of UPRNs (properties) within the cluster that are owner occupied",
        "perc_tenure_rental_(private)": "Percentage of UPRNs (properties) within the cluster that are private rental",
        "perc_tenure_rental_(social)": "Percentage of UPRNs (properties) within the cluster that are social rental",
        "perc_tenure_null": "Percentage of UPRNs (properties) within the cluster with unknown tenure",
        "current_energy_rating_a": "Number of UPRNs (properties) within the cluster with energy rating A",
        "current_energy_rating_b": "Number of UPRNs (properties) within the cluster with energy rating B",
        "current_energy_rating_c": "Number of UPRNs (properties) within the cluster with energy rating C",
        "current_energy_rating_d": "Number of UPRNs (properties) within the cluster with energy rating D",
        "current_energy_rating_e": "Number of UPRNs (properties) within the cluster with energy rating E",
        "current_energy_rating_f": "Number of UPRNs (properties) within the cluster with energy rating F",
        "current_energy_rating_g": "Number of UPRNs (properties) within the cluster with energy rating G",
        "current_energy_rating_null": "Number of UPRNs (properties) within the cluster with unknown energy rating",
        "median_estimated_energy_consumption_12_months_kwh_per_m2": "Median estimated energy consumption in 12 months (in kWh/m2) of properties within the cluster",
        "median_outdoor_space_m2": "Median of the maximum contigous outdoor space in m2 of properties within the cluster",
        "in_hn_zone": "Whether any properties within the cluster are in DESNZ heat network zones. You can refer to: https://www.gov.uk/government/publications/heat-network-zone-opportunity-reports",
        "in_city_centre": "Whether any properties within the cluster are in 'city centres' as per the Spatial Signatures Framework",
        f"within_{COASTLINE_DISTANCE_THRESHOLD_M}m_coastline": f"Whether any properties within the cluster are within {COASTLINE_DISTANCE_THRESHOLD_M}m of the coastline",
        "in_protected_area": "Whether any properties within the cluster are in conservation areas for England and Wales, or Scottish World Heritage sites.",
        f"within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load": f"Whether the cluster is within {ANCHOR_LOAD_RADIUS}m from an anchor load (a location with high electricity demand, e.g. a school or hospital). See a full list of anchor load types in: ",
    },
}
