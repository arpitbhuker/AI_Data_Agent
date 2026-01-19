import pandas as pd


def generate_eda(df: pd.DataFrame):
    """
    Returns:
    - eda_report: dict (safe for memory + llm)
    - eda_tables: dict of DataFrames (for Streamlit UI)
    """

    eda_report = {}
    eda_tables = {}

    # ---------------- BASIC ----------------
    eda_report["shape"] = df.shape

    # ---------------- MISSING ----------------
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)

    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_%": missing_pct
    })
    missing_table = missing_table[missing_table["missing_count"] > 0]

    eda_tables["missing_table"] = missing_table

    eda_report["missing"] = df.isna().sum().to_dict()

    # ---------------- DTYPES ----------------
    dtypes_table = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "unique_values": [df[c].nunique(dropna=True) for c in df.columns]
    })

    eda_tables["dtypes_table"] = dtypes_table

    eda_report["dtypes"] = df.dtypes.astype(str).to_dict()

    # ---------------- NUMERIC SUMMARY ----------------
    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().T.round(2)
        numeric_summary["missing_count"] = numeric_df.isna().sum()
        numeric_summary["missing_%"] = (numeric_df.isna().mean() * 100).round(2)

        eda_tables["numeric_summary_table"] = numeric_summary

        # Still store a dict version for memory/llm
        eda_report["numeric_summary"] = numeric_summary.to_dict()

        # ---------------- CORRELATION MATRIX ----------------
        corr = numeric_df.corr().round(2)
        eda_tables["correlation_table"] = corr
        eda_report["correlation_matrix"] = corr.to_dict()

        # ---------------- TOP CORRELATIONS ----------------
        corr_abs = corr.abs()
        stacked = corr_abs.unstack().sort_values(ascending=False)

        # remove self-correlation
        stacked = stacked[stacked < 1]

        top_pairs = []
        used = set()

        for (a, b), value in stacked.items():
            if (b, a) in used:
                continue
            used.add((a, b))
            top_pairs.append((a, b, float(value)))
            if len(top_pairs) == 10:
                break

        top_corr_table = pd.DataFrame(top_pairs, columns=["feature_1", "feature_2", "abs_corr"])
        eda_tables["top_correlations_table"] = top_corr_table

        eda_report["top_correlations"] = [
            f"{row.feature_1} vs {row.feature_2}: {row.abs_corr}"
            for _, row in top_corr_table.iterrows()
        ]

    return eda_report, eda_tables


def target_eda(df, target):
    insights = []

    if target not in df.columns:
        return insights

    if df[target].nunique() <= 10:
        counts = df[target].value_counts(normalize=True) * 100
        insights.append(
            f"Target '{target}' is categorical with class distribution: "
            f"{counts.round(2).to_dict()}"
        )
    else:
        stats = df[target].describe().round(2).to_dict()
        insights.append(
            f"Target '{target}' is numeric with distribution stats: {stats}"
        )

    return insights