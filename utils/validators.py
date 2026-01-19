import pandas as pd

def validate_dataframe(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Uploaded dataset is empty")

    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 columns")

    return True
