def generate_insights(eda_report, cleaning_stats):
    insights = []

    if cleaning_stats["duplicates_removed"] > 0:
        insights.append(
            f"Dataset contained duplicate records, indicating potential data collection redundancy."
        )

    if len(cleaning_stats["missing_values_filled"]) > 0:
        insights.append(
            "Missing values were present, suggesting incomplete survey responses or data capture issues."
        )

    return insights
