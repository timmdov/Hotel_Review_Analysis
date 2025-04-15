import pandas as pd

from config import LABELED_PATH, COLUMNS_TO_KEEP_FOR_MODEL, FINAL_PATH

def prepare_dataset_for_model(input_path, output_path):
    df = pd.read_csv(input_path)
    df_slim = df[COLUMNS_TO_KEEP_FOR_MODEL]
    df_slim.to_csv(output_path, index=False)

if __name__ == "__main__":
    prepare_dataset_for_model(
        input_path=LABELED_PATH,
        output_path=FINAL_PATH,
    )