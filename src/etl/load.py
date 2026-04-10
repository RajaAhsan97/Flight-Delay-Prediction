# ====================================
# Load data from csv file
# ===================================

import pandas as pd
import os

BASE_DIR = "data/cleaned" 

def save_data(df: pd.DataFrame, output_path: str) -> str:
    os.makedirs(BASE_DIR, exist_ok=True)
    save_path = os.path.join(BASE_DIR, output_path)

    df.to_csv(save_path, index=False)
    print(f"Data saved to {output_path}, shape: {df.shape}")
    return output_path

# Example usage
if __name__ == "__main__":
    #from src.etl.extract import load_data
    from src.etl.extract import download_data    
    from src.etl.transform import clean_data
    
    #raw_df = load_data("data/raw/T_ONTIME_REPORTING.csv")
    raw_df = download_data(2023, 1)
    clean_df, _ = clean_data(raw_df)
    print("DataFrame Columns: ", clean_df.columns.tolist())
    save_data(clean_df, "T_ONTIME_REPORTING_cleaned.csv")



# run this using
# python -m src.etl.load

