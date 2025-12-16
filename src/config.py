import pandas as pd
import os

def load_csv(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def load_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """Load data from an Excel file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_excel(file_path, sheet_name=sheet_name)

