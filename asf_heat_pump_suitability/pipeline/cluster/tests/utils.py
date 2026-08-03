def assign_bool_domestic_status(b_id: str, non_domestic: list) -> bool:
    """
    Returns False for specified non-domestic codes.

    Args:
        b_id (str): test building ID
        non_domestic (list): list of test building IDs to be labelled as non-domestic buildings

    Returns:
        bool: returns False if building ID is non-domestic and True if it is domestic
    """
    # Extract test building ID prefix (e.g. 'B03')
    prefix = b_id.split("_")[0].upper()
    return prefix not in non_domestic


def assign_str_tech_type(b_id: str, tech_mapping: dict) -> str:
    """
    Maps technology types based on building ID prefixes.

    Args:
        b_id (str): test building ID
        tech_mapping (dict): building ID prefix (e.g. BO3) to tech type mapping

    Returns:
        str: tech type assigned to test building ID
    """
    prefix = b_id.split("_")[0].upper()
    return tech_mapping.get(prefix, None)
