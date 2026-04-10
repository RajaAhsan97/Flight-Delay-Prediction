# ====================================
# Transform data from csv file
# ====================================

# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import LabelEncoder

# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     # Data Cleaning ----------------------------

#     # 1. Drop rows without target
#     df = df.dropna(subset=['ARR_DELAY', 'DEP_DELAY'])
#     print(f'After dropping null ARR/DEP_DELAY: {df.shape}')

#     # 2. Remove extreme outliers
#     df = df[(df['ARR_DELAY'] > -120) & (df['ARR_DELAY'] < 600)]
#     df = df[(df['DEP_DELAY'] > -120) & (df['DEP_DELAY'] < 600)]
#     print(f'After outlier removal: {df.shape}')

#     # 3. Fill delay columns with 0 (NaN means no delay of that type)
#     df['CARRIER_DELAY'] = df['CARRIER_DELAY'].fillna(0)
#     df['WEATHER_DELAY'] = df['WEATHER_DELAY'].fillna(0)

#     # 4. Fill TAXI columns with median
#     df['TAXI_OUT'] = df['TAXI_OUT'].fillna(df['TAXI_OUT'].median())
#     df['TAXI_IN']  = df['TAXI_IN'].fillna(df['TAXI_IN'].median())

#     print(f'\n Cleaned shape: {df.shape}')
#     print(f'Remaining nulls: {df.isnull().sum().sum()}')

#     # Feature Engineering ----------------------------

#     # Parse FL_DATE
#     df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
#     df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1   # 1=Mon … 7=Sun
#     df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
 
#     # Weekend flag
#     df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)

#     # Distance buckets
#     df['DIST_BUCKET'] = pd.cut(
#         df['DISTANCE'],
#         bins=[0, 500, 1000, 1500, 2000, 10000],
#         labels=['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
#     )

#     # Route
#     df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']

#     # Encode categoricals
#     le = {}
#     for col in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DIST_BUCKET', 'ROUTE']:
#         le[col] = LabelEncoder()
#         df[col + '_ENC'] = le[col].fit_transform(df[col].astype(str))

#     # Classification target
#     df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)

#     print(' Features engineered!')
#     print(df.shape)

#     return df, le

# # Example usage
# if __name__ == "__main__":
#     from src.etl.extract import load_data
#     raw_df = load_data("data/raw/T_ONTIME_REPORTING.csv")
#     clean_df, _ = clean_data(raw_df)
#     print(clean_df.head())

# # run this using
# # python -m src.etl.transform

# ====================================
# Transform data from url file
# ===================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# These columns are required to train the model
REQUIRED_COLUMNS = [
    "YEAR", "MONTH", "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN", "DEST",
    "DEP_DELAY", "TAXI_OUT", "TAXI_IN",
    "ARR_DELAY", "DISTANCE",
    "CARRIER_DELAY", "WEATHER_DELAY"
]

# The extracted dataset from BTS url contains different column names 
# But the column names I have used to transform and clean datset are different
# So i map the Extracted dataset columns
COLUMN_MAPPING = {
    "Year": "YEAR",                   
    "Month": "MONTH",            
    "FlightDate": "FL_DATE",          
    "Reporting_Airline": "OP_UNIQUE_CARRIER",    
    "Origin": "ORIGIN",
    "Dest": "DEST",             
    "DepDelay": "DEP_DELAY",            
    "TaxiOut": "TAXI_OUT",             
    "TaxiIn": "TAXI_IN",              
    "ArrDelay": "ARR_DELAY",          
    "Distance": "DISTANCE",           
    "CarrierDelay": "CARRIER_DELAY",        
    "WeatherDelay": "WEATHER_DELAY"        
}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    df = df[REQUIRED_COLUMNS]

    # Data Cleaning ----------------------------

    # 1. Drop rows without target
    df = df.dropna(subset=['ARR_DELAY', 'DEP_DELAY'])
    print(f'After dropping null ARR/DEP_DELAY: {df.shape}')

    # 2. Remove extreme outliers
    df = df[(df['ARR_DELAY'] > -120) & (df['ARR_DELAY'] < 600)]
    df = df[(df['DEP_DELAY'] > -120) & (df['DEP_DELAY'] < 600)]
    print(f'After outlier removal: {df.shape}')

    # 3. Fill delay columns with 0 (NaN means no delay of that type)
    df['CARRIER_DELAY'] = df['CARRIER_DELAY'].fillna(0)
    df['WEATHER_DELAY'] = df['WEATHER_DELAY'].fillna(0)

    # 4. Fill TAXI columns with median
    df['TAXI_OUT'] = df['TAXI_OUT'].fillna(df['TAXI_OUT'].median())
    df['TAXI_IN']  = df['TAXI_IN'].fillna(df['TAXI_IN'].median())

    print(f'\n Cleaned shape: {df.shape}')
    print(f'Remaining nulls: {df.isnull().sum().sum()}')

    # Feature Engineering ----------------------------

    # Parse FL_DATE
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek + 1   # 1=Mon … 7=Sun
    df['DAY_OF_MONTH'] = df['FL_DATE'].dt.day
 
    # Weekend flag
    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin([6, 7]).astype(int)

    # Distance buckets
    df['DIST_BUCKET'] = pd.cut(
        df['DISTANCE'],
        bins=[0, 500, 1000, 1500, 2000, 10000],
        labels=['VeryShort', 'Short', 'Medium', 'Long', 'VeryLong']
    )

    # Route
    df['ROUTE'] = df['ORIGIN'] + '_' + df['DEST']

    # Encode categoricals
    le = {}
    for col in ['OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST', 'DIST_BUCKET', 'ROUTE']:
        le[col] = LabelEncoder()
        df[col + '_ENC'] = le[col].fit_transform(df[col].astype(str))

    # Classification target
    df['IS_DELAYED'] = (df['ARR_DELAY'] > 15).astype(int)

    print(' Features engineered!')
    print(df.shape)

    return df, le

# Example usage
if __name__ == "__main__":
    from src.etl.extract import download_data
    raw_df = download_data(2023, 1)
    clean_df, _ = clean_data(raw_df)
    print(clean_df.head())
