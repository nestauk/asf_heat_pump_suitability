def assign_bool_domestic_status(b_id: str, non_domestic: list) -> bool:
    """
    Returns False for specified non-domestic codes.

    Args:
        b_id (str): test building ID, e.g. 'B01', or 'b03_cluster1' where 'b03' is the ID prefix and 'cluster1' is the descriptor
        non_domestic (list): list of test building IDs to be labelled as non-domestic buildings

    Returns:
        bool: returns False if building ID is non-domestic and True if it is domestic
    """
    # Extract test building ID prefix (e.g. 'B03' from 'b03_cluster1')
    prefix = b_id.split("_")[0].upper()
    return prefix not in non_domestic


def assign_str_tech_type(b_id: str, tech_mapping: dict) -> str:
    """
    Maps technology types based on building ID prefixes.

    Args:
        b_id (str): test building ID, e.g. 'B01', or 'b03_cluster1' where 'b03' is the ID prefix and 'cluster1' is the descriptor
        tech_mapping (dict): building ID prefix (e.g. BO3) to tech type mapping

    Returns:
        str: tech type assigned to test building ID
    """
    # Extract test building ID prefix (e.g. 'B03' from 'b03_cluster1')
    prefix = b_id.split("_")[0].upper()
    return tech_mapping.get(prefix, None)
