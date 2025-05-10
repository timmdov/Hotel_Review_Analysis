import pandas as pd

from src.utils.config.paths import RAW_PATH, STEP1_FILTERED_COLUMNS, COLUMNS_TO_KEEP
from src.utils.io.logger import get_logger

logger = get_logger(__name__)


def remove_unnecessary_columns(input_path, output_path, columns_to_keep):
    """
        Reads a dataset from `input_path`, keeps only the specified columns,
        and saves the filtered dataset to `output_path`.

        Parameters:
        - input_path (str): Path to the original CSV file.
        - output_path (str): Path where the filtered CSV will be saved.
        - columns_to_keep (list): List of column names to keep in the new file.
        """
    try:
        df = pd.read_csv(input_path)
        df_filtered = df[columns_to_keep]
        df_filtered.to_csv(output_path, index=False)
        logger.info(f"Saved filtered dataset to: {output_path}")
    except Exception as e:
        logger.info(f"Error: {e}")


def run_filter_columns():
    remove_unnecessary_columns(
        input_path=RAW_PATH,
        output_path=STEP1_FILTERED_COLUMNS,
        columns_to_keep=COLUMNS_TO_KEEP,
    )


if __name__ == "__main__":
    run_filter_columns()
