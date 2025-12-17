# train_test_split.py
from sklearn.model_selection import train_test_split
import pandas as pd

#split into train test and validation
#0.7 train, 0.15 val, 0.15 test
def split_data(df: pd.DataFrame, target_column: str, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Calculate the proportion of validation size with respect to the remaining data
    val_proportion = val_size / (1 - test_size)
    
    # Now split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_proportion, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test