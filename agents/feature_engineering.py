import pandas as pd


def engineer_features(df: pd.DataFrame):
    df = df.copy()
    report = []

    # --- Detect datetime columns & extract features ---
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                df[col] = parsed
                report.append(f"Parsed '{col}' as datetime.")
            except Exception:
                continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df = df.drop(columns=[col])
            report.append(f"Extracted year/month/day features from datetime column '{col}'.")

    # --- Encode binary categoricals safely ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        nunique = df[col].nunique()

        if nunique == 2:
            df[col] = df[col].astype("category").cat.codes
            report.append(f"Binary-encoded '{col}' for modeling compatibility.")

        # drop high-cardinality categoricals (only if too huge)
        elif nunique > 50:
            df = df.drop(columns=[col])
            report.append(f"Dropped high-cardinality column '{col}' (too many unique categories).")

    # --- Age binning remains but safe ---
    if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 18, 35, 50, 65, 120],
            labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"]
        )
        report.append("Created 'age_group' feature to capture non-linear age impact.")

    return df, report
