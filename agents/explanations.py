def explain_eda(eda):
    explanations = []

    rows, cols = eda["shape"]
    explanations.append(
        f"The dataset contains {rows} rows and {cols} columns."
    )

    for col, missing in eda["missing"].items():
        if missing > 0:
            explanations.append(
                f"Column '{col}' has missing values which were addressed during cleaning."
            )

    return explanations
