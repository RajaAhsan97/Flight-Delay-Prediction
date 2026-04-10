# ====================================
# Fetch data from csv file
# ====================================
# import pandas as pd
# import os

# def load_data(file_path: str) -> pd.DataFrame:
#     """
#     Load raw data from CSV.
#     """
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"{file_path} does not exist")
    
#     df = pd.read_csv(file_path)
#     print(f"Data loaded from {file_path}, shape: {df.shape}")
#     return df

# if __name__ == "__main__":
#     df = load_data("data/raw/T_ONTIME_REPORTING.csv")
#     print(df.head())

# # run this using
# # python -m src.etl.extract

# ====================================
# Fetch data from csv file
# ===================================

import requests
import zipfile
import os

import pandas as pd

BASE_DIR = "data/raw"

def get_next_version():
    if not os.path.exists(BASE_DIR):
        return "v1"

    versions = [d for d in os.listdir(BASE_DIR) if d.startswith("v")]

    if not versions:
        return "v1"

    version_numbers = [int(v[1:]) for v in versions]
    next_version = max(version_numbers) + 1

    return f"v{next_version}"


def download_data(year, month):
    #url = f"https://transtats.bts.gov/PREZIP/On_Time_On_Time_Performance_{year}_{month}.zip"

    url = f"https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"

    version = get_next_version()
    save_path = os.path.join(BASE_DIR, version)

    os.makedirs(save_path, exist_ok=True)

    zip_path = os.path.join(save_path, f"data_{year}_{month}.zip")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200 and "zip" in response.headers.get("Content-Type", ""):
        with open(zip_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {zip_path}")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_path)

        print(f"Extracted in {save_path}")

    else:
        print("Download failed")

    files = os.listdir(save_path)

    # Find CSV file
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        raise Exception("No CSV file found!")

    file_path = os.path.join(save_path, csv_files[0])

    print(f"Loading file: {file_path}")

    df = pd.read_csv(file_path)

    return df

if __name__ == "__main__":
    df = download_data(2023, 1)