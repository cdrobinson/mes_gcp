"""Big Query client for interacting with Google BigQuery."""

import logging
from typing import Iterator, List, Optional
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Conflict, BadRequest
import pandas as pd

from utils.retry import RetryableClient, retry_with_backoff

logger = logging.getLogger(__name__)

class BigQueryClient(RetryableClient):
    """Client for interacting with Google BigQuery."""

    def __init__(self, project_id: str, location: str, **retry_kwargs):
        """
        Initialise BigQuery client.

        Args:
            project_id: GCP project ID.
            location: GCP location.
            **retry_kwargs: Optional arguments for retry config.
        """
        super().__init__(**retry_kwargs)
        self.project_id = project_id
        self.location = location
        self.client = bigquery.Client(project=project_id, location=location)

    @retry_with_backoff(max_attempts=3)
    def create_dataset(self, dataset_name: str, exists_ok: bool = True) -> bigquery.Dataset:
        """
        Create a BigQuery dataset if it does not exist.

        Args:
            dataset_name: Name of the dataset.
            exists_ok: If True, don't raise if exists.

        Returns:
            The created or existing dataset.
        """
        dataset_id = f"{self.project_id}.{dataset_name}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = self.location

        try:
            dataset = self.client.create_dataset(dataset, exists_ok=exists_ok)
            logger.info(f"Dataset {dataset_id} created.")
        except Conflict:
            if exists_ok:
                dataset = self.client.get_dataset(dataset_id)
                logger.info(f"Dataset {dataset_id} already exists.")
            else:
                logger.error(f"Conflict creating dataset {dataset_id}.")
                raise
        return dataset

    @retry_with_backoff(max_attempts=3)
    def create_table(
        self,
        dataset_name: str,
        table_name: str,
        schema: List[bigquery.SchemaField],
        exists_ok: bool = True,
        time_partitioning: Optional[bigquery.TimePartitioning] = None,
        clustering_fields: Optional[List[str]] = None,
    ) -> bigquery.Table:
        """
        Create a BigQuery table if it does not exist.

        Args:
            dataset_name: Name of the dataset.
            table_name: Name of the table.
            schema: Table schema.
            exists_ok: If True, don't raise if exists.
            time_partitioning: Optional time partitioning.
            clustering_fields: Optional clustering fields.

        Returns:
            The created or existing table.
        """
        table_id = f"{self.project_id}.{dataset_name}.{table_name}"
        table = bigquery.Table(table_id, schema=schema)
        if time_partitioning:
            table.time_partitioning = time_partitioning
        if clustering_fields:
            table.clustering_fields = clustering_fields

        try:
            table = self.client.create_table(table, exists_ok=exists_ok)
            logger.info(f"Table {table_id} created.")
        except Conflict:
            if exists_ok:
                table = self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists.")
            else:
                logger.error(f"Conflict creating table {table_id}.")
                raise
        return table

    @retry_with_backoff(max_attempts=3)
    def insert_rows_from_dataframe(
        self,
        dataset_name: str,
        table_name: str,
        df: pd.DataFrame,
        ignore_unknown_values: bool = False,
        skip_invalid_rows: bool = False
    ) -> None:
        """
        Insert rows into a BigQuery table from a pandas DataFrame.

        Args:
            dataset_name: Name of dataset.
            table_name: Name of table.
            df: DataFrame to insert.
            ignore_unknown_values: If True, ignore unknown fields.
            skip_invalid_rows: If True, skip invalid rows.

        Raises:
            RuntimeError: If any insert errors occur.
        """
        table_id = f"{self.project_id}.{dataset_name}.{table_name}"

        try:
            table = self.client.get_table(table_id)

            errors = self.client.insert_rows_from_dataframe(
                table=table,
                dataframe=df,
                ignore_unknown_values=ignore_unknown_values,
                skip_invalid_rows=skip_invalid_rows
            )
            if any(errors):
                logger.error(f"Error inserting DataFrame rows into {table_id}: {errors}")
                raise RuntimeError(f"Insert errors: {errors}")
            logger.info(f"Inserted {len(df)} rows from DataFrame into {table_id}.")
        except BadRequest as e:
            logger.error(f"Bad request inserting DataFrame into {table_id}: {e}")
            raise

    @retry_with_backoff(max_attempts=3)
    def query(self, sql: str) -> Iterator[bigquery.table.Row]:
        """
        Run a SQL query.

        Args:
            sql: SQL string.

        Returns:
            Iterator over rows.
        """
        try:
            query_job = self.client.query(sql)
            logger.info(f"Running query: {sql}")
            return query_job.result()
        except BadRequest as e:
            logger.error(f"Query failed: {e}")
            raise

    @retry_with_backoff(max_attempts=3)
    def get_table_schema(self, dataset_name: str, table_name: str) -> List[bigquery.SchemaField]:
        """
        Get the schema for a table.

        Args:
            dataset_name: Dataset name.
            table_name: Table name.

        Returns:
            List of SchemaField.
        """
        table_id = f"{self.project_id}.{dataset_name}.{table_name}"
        try:
            table = self.client.get_table(table_id)
            logger.info(f"Fetched schema for {table_id}.")
            return table.schema
        except NotFound:
            logger.error(f"Table {table_id} not found.")
            raise
