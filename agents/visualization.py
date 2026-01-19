import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import textwrap

FIG_SIZE = (5, 3.5)


# ----------------------------- Helpers -----------------------------

def _has_outliers(series: pd.Series) -> bool:
    series = series.dropna()
    if series.empty:
        return False

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return False
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).any()


def _winsorize_series(series: pd.Series, lower_q=0.01, upper_q=0.99):
    """Cap extreme values just for visualization (not for model)."""
    s = series.copy()
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def _shorten_labels(values, max_len=12):
    """
    Shorten long category names: 'premium unleaded (recommended)' -> 'premium unlea…'
    """
    out = []
    for v in values:
        s = str(v)
        s = s.replace("_", " ").strip()
        if len(s) > max_len:
            s = s[: max_len - 1] + "…"
        out.append(s)
    return out


def _group_rare_categories(df: pd.DataFrame, col: str, top_n=8):
    """
    Keep only top_n categories by frequency, rest -> 'Other'
    This avoids axis clutter.
    """
    df = df.copy()
    counts = df[col].value_counts(dropna=False)
    keep = counts.head(top_n).index
    df[col] = df[col].where(df[col].isin(keep), "Other")
    return df


def _pick_top_numeric(df: pd.DataFrame, k: int = 3):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return []
    variances = df[numeric_cols].var().sort_values(ascending=False)
    return variances.head(k).index.tolist()


def _pick_top_categorical(df: pd.DataFrame, k: int = 2):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # remove crazy cardinality categoricals
    cat_cols = [c for c in cat_cols if 2 <= df[c].nunique() <= 25]

    # prioritize medium-cardinality (most informative)
    cat_cols = sorted(cat_cols, key=lambda c: df[c].nunique(), reverse=True)
    return cat_cols[:k]


def _pick_top_corr_pair(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    pair = np.unravel_index(np.argmax(corr.values), corr.shape)
    col1 = corr.columns[pair[0]]
    col2 = corr.columns[pair[1]]

    # if correlation is meaningless, skip
    if corr.loc[col1, col2] < 0.25:
        return None

    return col1, col2


# ----------------------------- Main -----------------------------

def auto_visualize(df, profile=None):
    plots = []
    df = df.copy()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        return plots

    # -------- 1) Numeric Distributions (Top 2 by variance) --------
    top_num = _pick_top_numeric(df, k=2)

    for col in top_num:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution: {col}")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plots.append(fig)

    # -------- 2) Correlation heatmap (Top 10 numeric max) --------
    if len(numeric_cols) >= 2:
        # limit heatmap size (else unreadable)
        heat_cols = numeric_cols[:10]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        corr = df[heat_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap (Top Numeric Features)")
        plt.tight_layout()
        plots.append(fig)

    # -------- 3) Scatter plot (Top correlated pair) --------
    pair = _pick_top_corr_pair(df)
    if pair:
        x_col, y_col = pair
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.6, ax=ax)
        ax.set_title(f"Relationship: {x_col} vs {y_col}")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plots.append(fig)

    # -------- 4) Categorical vs Numeric (Smart readability) --------
    top_cat = _pick_top_categorical(df, k=2)

    if top_cat and top_num:
        comparisons = 0

        for cat in top_cat:
            # Reduce clutter
            df_plot = _group_rare_categories(df, cat, top_n=8)

            # decide orientation based on max label length
            label_lengths = df_plot[cat].astype(str).map(len)
            long_labels = label_lengths.max() > 12

            for num in top_num:
                if comparisons >= 2:
                    break

                fig, ax = plt.subplots(figsize=(5, 3.8))

                # make values visible
                y_series = df_plot[num]
                if _has_outliers(y_series):
                    # hide fliers + winsorize for visibility
                    y_plot = _winsorize_series(y_series)
                    plot_title_suffix = " (Outliers handled)"
                else:
                    y_plot = y_series
                    plot_title_suffix = ""

                tmp = df_plot.copy()
                tmp[num] = y_plot

                # chart type decision
                if long_labels:
                    # Horizontal = readable
                    sns.boxplot(y=tmp[cat], x=tmp[num], showfliers=False, ax=ax)
                    ax.set_title(f"{num} by {cat}{plot_title_suffix}")
                    ax.set_xlabel(num)
                    ax.set_ylabel(cat)
                else:
                    # Vertical plot with rotated labels
                    sns.boxplot(x=tmp[cat], y=tmp[num], showfliers=False, ax=ax)
                    ax.set_title(f"{num} by {cat}{plot_title_suffix}")
                    ax.set_xlabel(cat)
                    ax.set_ylabel(num)

                    # rotate + shorten tick labels
                    tick_labels = _shorten_labels(ax.get_xticklabels(), max_len=12)
                    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

                ax.grid(alpha=0.15)
                plt.tight_layout()
                plots.append(fig)
                comparisons += 1

    return plots
