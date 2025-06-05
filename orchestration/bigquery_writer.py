import pandas as pd
from google.cloud import bigquery
from core.gcp_client import get_bigquery_client, get_bq_dataset_id
from typing import List, Dict, Any
import json # Added for robust serialization

class BigQueryWriter:
    """Writes experiment results to BigQuery."""

    def __init__(self):
        self.client = get_bigquery_client()
        self.dataset_id = get_bq_dataset_id()

    def _get_full_table_id(self, table_id: str) -> str:
        """Constructs the full BigQuery table ID."""
        return f"{self.client.project}.{self.dataset_id}.{table_id}"

    def _ensure_serializable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts non-BQ-friendly types (like nested dicts/lists) to JSON strings."""
        for col in df.columns:
            # Check if any cell in the column is a dict or list
            if any(isinstance(val, (dict, list)) for val in df[col]):
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
        return df

    def write_results(self, results: List[Dict[str, Any]], table_id: str):
        """
        Writes a list of result dictionaries to the specified BigQuery table.
        """
        if not results:
            print("No results to write to BigQuery.")
            return

        full_table_id = self._get_full_table_id(table_id)
        df = pd.DataFrame(results)
        
        # Ensure data is serializable for BigQuery
        df = self._ensure_serializable(df.copy()) # Use .copy() to avoid SettingWithCopyWarning

        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            # schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION] # autodetect handles this
        )

        try:
            load_job = self.client.load_table_from_dataframe(
                df, full_table_id, job_config=job_config
            )
            load_job.result()
            print(f"Loaded {load_job.output_rows} rows to {full_table_id}.")
        except Exception as e:
            print(f"Error writing to BigQuery table {full_table_id}: {e}")
            raise