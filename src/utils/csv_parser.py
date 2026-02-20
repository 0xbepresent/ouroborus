"""
CSV parsing utilities for handling CodeQL CSV files.

This module provides utilities for parsing CSV rows that may contain commas
inside quoted fields, which requires regex-based parsing rather than
simple string splitting.
"""

import re
from typing import Dict, List


# Regex pattern for splitting CSV rows while handling commas inside quoted fields
CSV_SPLIT_PATTERN = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')


def parse_csv_row(row: str, keys: List[str]) -> Dict[str, str]:
    """
    Parse a CSV row into a dictionary using regex to handle commas inside quotes.

    This function is designed for line-by-line CSV parsing where rows may contain
    commas within quoted fields. It uses a regex pattern to split on commas that
    are not inside quotes.

    Args:
        row (str): The raw CSV row string (may include newline).
        keys (List[str]): List of field names to map the split values to.

    Returns:
        Dict[str, str]: Dictionary mapping keys to CSV field values.
        
    """
    row_split = CSV_SPLIT_PATTERN.split(row)
    return dict(zip(keys, row_split))


