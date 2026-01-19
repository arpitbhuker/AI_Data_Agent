import pandas as pd
from agents.cleaning import clean_data

def test_cleaning_removes_duplicates():
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    cleaned, report = clean_data(df)
    assert len(cleaned) == 1
