import pandas as pd
from typing import List, Dict, Any

def flatten_results_for_bq(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flattens nested dictionaries within a list of results for BigQuery compatibility.
    BigQuery prefers flat structures or JSON strings for nested data.
    This function converts nested dicts into top-level keys with prefixed names.
    Example: {"metadata": {"model": "gemini"}} becomes {"metadata_model": "gemini"}
    """
    flat_results = []
    for record in results:
        flat_record = {}
        for key, value in record.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_record[f"{key}_{sub_key}"] = sub_value
            else:
                flat_record[key] = value
        flat_results.append(flat_record)
    return flat_results

def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts a list of result dictionaries to a Pandas DataFrame."""
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)