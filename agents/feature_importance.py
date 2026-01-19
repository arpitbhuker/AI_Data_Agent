import pandas as pd

def feature_importance(df, target):
    importance = []

    if target not in df.columns:
        return importance

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols = [c for c in numeric_cols if c != target]

    if not numeric_cols:
        return importance

    correlations = (
        df[numeric_cols + [target]]
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    for feature, score in correlations.items():
        importance.append(
            f"Feature '{feature}' shows correlation strength {round(score, 3)} "
            f"with target '{target}', indicating potential predictive relevance."
        )

    return importance
