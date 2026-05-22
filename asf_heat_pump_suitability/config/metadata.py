from datetime import datetime
from asf_heat_pump_suitability import config

ANCHOR_LOAD_RADIUS = config["constant"]["clustering"]["anchor_load_radius"]
metadata = {
    "License": "This dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.",
    "Date of creation": datetime.now().strftime("%Y-%m-%d"),
    "Data sources and attributions": "https://github.com/nestauk/asf_heat_pump_suitability/tree/dev/asf_heat_pump_suitability/config#readme",
    "Variable descriptions": {
        "cluster_id": "Unique identifier for each cluster",
        "n_UPRNs": "Number of domestic UPRNs (properties) within the cluster",
        "n_uprns_in_listed_building": "Number of UPRNs (properties) within the cluster that are in listed buildings",
        "n_uprns_solar_pv": "Number of UPRNs (properties) within the cluster that have solar PV",
        "n_uprns_off_gas": "Number of UPRNs (properties) within the cluster that are off-gas",
        "median_estimated_energy_consumption_12_months_kwh_per_m2": "Median estimated energy consumption in 12 months (in kWh/m2) of properties within the cluster",
        "median_outdoor_space_m2": "Median of the maximum contigous outdoor space in m2 of properties within the cluster",
        "in_hn_zone": "Whether any properties within the cluster are in DESNZ heat network zones. You can refer to: https://www.gov.uk/government/publications/heat-network-zone-opportunity-reports",
        "in_city_centre": "Whether any properties within the cluster are in 'city centres' as per the Spatial Signatures Framework",
        "within_1500m_coastline": "Whether any properties within the cluster are within 1500m of the coastline",
        "in_protected_area": "Whether any properties within the cluster are in conservation areas for England and Wales, or Scottish World Heritage sites.",
        "within_{ANCHOR_LOAD_RADIUS}m_from_anchor_load": f"Whether the cluster is within {ANCHOR_LOAD_RADIUS}m from an anchor load (a location with high electricity demand, e.g. a school or hospital). See a full list of anchor load types in: ",
    },
}
