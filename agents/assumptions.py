def eda_assumptions(df, target=None):
    assumptions = []

    assumptions.append(
        "Assumes the dataset represents a single consistent population "
        "without major distribution shifts."
    )

    if target:
        assumptions.append(
            f"Assumes '{target}' is the outcome variable and is not leaked "
            "into feature columns."
        )

    assumptions.append(
        "Assumes missing values are Missing At Random (MAR) and can be imputed."
    )

    assumptions.append(
        "Assumes rows are independent observations (no time dependency unless stated)."
    )

    assumptions.append(
        "Outliers are treated as valid extreme behavior unless explicitly removed."
    )

    return assumptions
