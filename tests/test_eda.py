import pandas as pd
from agents.eda import generate_eda

def test_eda_structure():
    df = pd.DataFrame({"x": [1, 2, 3]})
    eda = generate_eda(df)
    assert "shape" in eda
