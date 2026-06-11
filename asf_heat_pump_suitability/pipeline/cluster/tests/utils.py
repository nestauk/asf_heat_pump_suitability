def assign_domestic_status(b_id: str, non_domestic: list):
    """Returns False for specified non-domestic codes."""
    # Extract prefix (e.g. 'B03') cleanly regardless of lowercase user input
    prefix = b_id.split("_")[0].upper()
    return prefix not in non_domestic


def assign_tech_type(b_id: str, tech_mapping: dict):
    """Maps technology types based on building ID prefixes."""
    prefix = b_id.split("_")[0].upper()
    return tech_mapping.get(prefix, None)
