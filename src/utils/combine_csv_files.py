"""
Script to combine multiple CSV files into a single CSV file.

This script reads all CSV files from the dataset directory,
concatenates them, and saves the result as a single CSV file.
Uses only standard library modules to avoid dependency issues.
"""

import os
import csv
from pathlib import Path

def combine_csv_files(input_dir: str, output_file: str) -> None:
    """
    Combines all CSV files in the input directory into a single CSV file.

    Parameters:
    - input_dir (str): Path to the directory containing CSV files
    - output_file (str): Path to save the combined CSV file
    """
    try:
        # Get all CSV files in the input directory
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files in {input_dir}")

        if not csv_files:
            print(f"No CSV files found in {input_dir}")
            return

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # Open the output file for writing
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            # Process the first file to get headers
            first_file = os.path.join(input_dir, csv_files[0])
            print(f"Reading headers from {first_file}")

            with open(first_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                headers = next(reader)  # Get headers from first file

                # Create CSV writer with the same headers
                writer = csv.writer(outfile)
                writer.writerow(headers)

                # Write the rest of the first file
                for row in reader:
                    writer.writerow(row)

            # Process the rest of the files
            for file in csv_files[1:]:
                file_path = os.path.join(input_dir, file)
                print(f"Reading {file_path}")

                with open(file_path, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    next(reader)  # Skip header row

                    # Write all rows to the output file
                    for row in reader:
                        writer.writerow(row)

                print(f"Added data from {file}")

        print(f"Combined CSV file saved to {output_file}")

    except Exception as e:
        print(f"Error combining CSV files: {e}")

if __name__ == "__main__":
    # Define input and output paths
    input_directory = Path(__file__).parent.parent / "dataset"
    output_file = Path(__file__).parent.parent / "dataset" / "raw_reviews.csv"

    # Combine CSV files
    combine_csv_files(
        input_dir=str(input_directory),
        output_file=str(output_file)
    )

    print("CSV combination complete")