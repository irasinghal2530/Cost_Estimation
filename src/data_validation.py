# data_inspection.py
import pandas as pd

def inspect_data(df: pd.DataFrame):
    print("=== DATA INSPECTION ===")
    print("Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\nBasic Statistics:\n", df.describe(include='all'))

def remove_duplicates(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    return df.drop_duplicates(keep=keep)
def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', value=None) -> pd.DataFrame:
    # try: 
        if strategy == 'mean':
            return df.fillna(df.mean())
        elif strategy == 'median':
            return df.fillna(df.median())
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        elif strategy == 'constant':
            if value is None:
                raise ValueError("Value must be provided for constant fill strategy.")
            return df.fillna(value)
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'constant'.")  
 