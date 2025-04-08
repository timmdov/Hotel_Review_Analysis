import pandas as pd
import os

def save_filtered_dataset(input_csv_path: str, selected_fields: list[str]):
    try:
        user_path = input("📁 Enter the directory where you want to save the new dataset (or leave blank for current folder): ").strip()
        if user_path == "":
            user_path = "."
        os.makedirs(user_path, exist_ok=True)
        df = pd.read_csv(input_csv_path)
        filtered_df = df[selected_fields]
        field_part = "_".join([field.replace(" ", "") for field in selected_fields])
        output_filename = os.path.join(user_path, f"dataset_{field_part}.csv")
        filtered_df.to_csv(output_filename, index=False)
        print(f" Saved filtered dataset to: {output_filename}")
    except Exception as e:
        print(f" Error: {e}")
