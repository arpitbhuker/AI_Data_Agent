import pandas as pd

def profile_dataset(df: pd.DataFrame):
    df = df.copy()

    profile = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "numeric_cols": [],
        "categorical_cols": [],
        "datetime_cols": [],
        "id_like_cols": [],
        "constant_cols": [],
        "high_null_cols": [],
        "recommended_drop_cols": []
    }

    # Identify datetimes
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                # datetime if at least 70% values parse
                if parsed.notna().mean() > 0.7:
                    profile["datetime_cols"].append(col)
            except Exception:
                pass

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    profile["numeric_cols"] = [c for c in numeric_cols if c not in profile["datetime_cols"]]
    profile["categorical_cols"] = [c for c in cat_cols if c not in profile["datetime_cols"]]

    # Identify constant columns
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            profile["constant_cols"].append(col)

    # High null columns (>40% missing)
    for col in df.columns:
        null_ratio = df[col].isna().mean()
        if null_ratio > 0.40:
            profile["high_null_cols"].append(col)

    # ID-like columns (unique ratio > 0.9)
    for col in df.columns:
        if df[col].nunique(dropna=True) / max(1, len(df)) > 0.90:
            profile["id_like_cols"].append(col)

    # Recommended columns to drop
    drop_cols = set(profile["constant_cols"] + profile["high_null_cols"] + profile["id_like_cols"])
    profile["recommended_drop_cols"] = list(drop_cols)

    return profile
