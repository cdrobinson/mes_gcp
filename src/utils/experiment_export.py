"""Utility functions for exporting experiment results to CSV."""

import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

def experiment_exporter(
    experiment_name: str,
    results_df: pd.DataFrame
) -> str:
    """
    Save a dataframe to the experiment_outputs directory as CSV.
    
    Args:
        experiment_name: Name of the experiment (used in filename)
        results_df: The dataframe to save
    
    Returns:
        str: Path to the saved CSV file
    
    Raises:
        IOError: If unable to save the file
    """
    project_root = Path(__file__).parent.parent.parent
    experiment_outputs_dir = project_root / "experiment_outputs"
    
    experiment_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp_str}.csv"
    csv_path = experiment_outputs_dir / filename
    
    try:
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved experiment results to: {csv_path}")
        return str(csv_path)
        
    except Exception as e:
        logger.error(f"Failed to save experiment '{experiment_name}': {e}")
        raise IOError(f"Unable to save dataframe: {e}")
