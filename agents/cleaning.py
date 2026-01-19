import pandas as pd


def clean_data(df: pd.DataFrame, profile: dict | None = None):
    report_stats = {
        "duplicates_removed": 0,
        "missing_values_filled": {},
        "dropped_columns": []
    }

    report_text = []
    df = df.copy()

    # Drop recommended columns from profiling
    if profile and profile.get("recommended_drop_cols"):
        drop_cols = [c for c in profile["recommended_drop_cols"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            report_stats["dropped_columns"] = drop_cols
            report_text.append(f"Dropped columns based on profiling: {drop_cols}")

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        report_stats["duplicates_removed"] = int(dup_count)
        report_text.append(f"Removed {int(dup_count)} duplicate rows.")

    # Missing values
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count == 0:
            continue

        if df[col].dtype == "object":
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)
            report_text.append(f"Filled {missing_count} missing values in '{col}' using mode.")
        else:
            df[col] = df[col].fillna(df[col].median())
            report_text.append(f"Filled {missing_count} missing values in '{col}' using median.")

        report_stats["missing_values_filled"][col] = missing_count

    return df, report_stats, report_text
