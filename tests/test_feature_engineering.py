import pandas as pd
from agents.feature_engineering import engineer_features

def test_feature_engineering_output():
    df = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
    engineered, report = engineer_features(df)
    assert engineered.isnull().sum().sum() == 0
