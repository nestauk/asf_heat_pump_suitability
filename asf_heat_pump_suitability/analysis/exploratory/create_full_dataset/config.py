"""
Config file with defaults used for feasibility scoring and suitability categorisation.
"""

# Features needed for feasibility scoring
features = [
    "owner_occupied",
    "in_high_income_decile",
    "on_gas",
    "in_listed_building",
    "not_in_listed_building",
    "in_conservation_area",
    "not_in_conservation_area",
    "social_housing",
    "flats",
    "on_communal_heating",
    "has_outdoor_space",
    "in_heat_network_zone",
    "close_to_anchor_loads",
    "close_to_city_centre",
]

# Expected technology types for feasibility scoring
expected_tech_types = {
    "individual_ashp",
    "collective_ashp",
    "sgl",
    "hn",
}

# Dictionary of weights for computing feasibility scores for each tech type
weights = {
    "individual_ashp": {
        "owner_occupied": 1,
        "in_high_income_decile": 1,
        "on_gas": 1,
        "not_in_listed_building": 1,
        "not_in_conservation_area": 1,
    },
    "collective_ashp": {
        "owner_occupied": 1,
        "in_high_income_decile": 1,
        "on_gas": 1,
        "not_in_listed_building": 1,
        "not_in_conservation_area": 1,
        "cluster_size": 1,
    },
    "sgl": {
        "social_housing": 1,
        "flats": 1,
        "on_communal_heating": 1,
        "has_outdoor_space": 1,
        "cluster_size": 1,
    },
    "hn": {
        "in_heat_network_zone": 1,
        "close_to_anchor_loads": 1,
        "close_to_city_centre": 1,
    },
}

# OAs in the city centre of Plymouth
# Considering "City Centre, Barbican and Sutton Harbour" MSOA as the city center
# OAs can be found using https://www.ons.gov.uk/explore-local-statistics/areas/E02003148-plymouth-027
city_centre_oas = {
    # OAs in LSOA Plymouth 027A
    "E00076029",
    "E00076042",
    "E00076047",
    "E00076048",
    # OAs in LSOA Plymouth 027B
    "E00076028",
    "E00172050",
    "E00181096",
    "E00181180",
    "E00181183",
    # OAs in LSOA Plymouth 027C
    "E00076542",
    "E00076546",
    "E00076552",
    "E00076556",
    "E00076558",
    "E00076560",
    "E00181118",
    "E00181147",
    # OAs in LSOA Plymouth 027E
    "E00076559",
    "E00076573",
    "E00076574",
    "E00076577",
    "E00076579",
    "E00076583",
    # OAs in LSOA Plymouth 027F
    "E00076553",
    "E00076564",
    "E00172052",
    "E00172065",
    "E00172069",
    # OAs in LSOA Plymouth 027G
    "E00076568",
    "E00076569",
    "E00076571",
    "E00172054",
    "E00172055",
    "E00172056",
}
