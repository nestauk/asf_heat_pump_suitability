"""
Config file with defaults used for feasibility scoring and suitability categorisation.
"""

# Features needed for feasibility scoring
features = [
    "owner_occupied",
    "high_income_decile",
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

# Dictionary of weights for computing feasibility scores for each tech type
weights = {
    "individual_ashp_feasibility": {
        "owner_occupied": 1,
        # "in_high_income_decile": 1,
        "on_gas": 1,
        "not_listed": 1,
        "not_in_conservation_area": 1,
    },
    "collective_ashp_feasibility": {
        "owner_occupied": 1,
        # "in_high_income_decile": 1,
        "on_gas": 1,
        "not_listed": 1,
        "not_in_conservation_area": 1,
        "cluster_size": 1,
    },
    "sgl_feasibility": {
        "social_housing": 1,
        "flats": 1,
        # "on_communal_heating": 1,
        "has_outdoor_space": 1,
        "cluster_size": 1,
    },
    "hn_feasibility": {
        "in_hn": 1,
        # "close_to_anchor_loads": 1,
        # "close_to_city_center": 1
    },
}
