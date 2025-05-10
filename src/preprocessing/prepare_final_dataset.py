import pandas as pd

from src.utils.config.paths import STEP7_LABELED, COLUMNS_TO_KEEP_FOR_MODEL, STEP8_MODEL_READY

def prepare_dataset_for_model(input_path, output_path):
    df = pd.read_csv(input_path)
    df_slim = df[COLUMNS_TO_KEEP_FOR_MODEL]
    df_slim.to_csv(output_path, index=False)

if __name__ == "__main__":
    prepare_dataset_for_model(
        input_path=STEP7_LABELED,
        output_path=STEP8_MODEL_READY,
    )