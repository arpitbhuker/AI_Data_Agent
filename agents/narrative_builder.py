import re
import pandas as pd


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, dict):
        return [f"{k}: {v}" for k, v in x.items()]
    return [str(x).strip()]


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9%.\s_-]", "", s)
    return s


def dedupe_sentences(sentences, similarity_threshold=0.90):
    out = []
    seen = set()

    for s in sentences:
        if not s:
            continue
        n = normalize_text(s)
        if not n or n in seen:
            continue

        tokens = set(n.split())
        dup = False
        for prev in out:
            prev_n = normalize_text(prev)
            prev_tokens = set(prev_n.split())
            if not prev_tokens:
                continue
            jacc = len(tokens & prev_tokens) / max(1, len(tokens | prev_tokens))
            if jacc >= similarity_threshold:
                dup = True
                break

        if not dup:
            out.append(s)
            seen.add(n)

    return out


def rank_insights(insights, top_k=8):
    def score(x: str) -> float:
        xl = x.lower()

        risk = sum(k in xl for k in [
            "missing", "outlier", "skew", "imbalance", "leak",
            "high correlation", "duplicate", "drop", "id-like", "constant"
        ])
        has_number = bool(re.search(r"\d", x))
        actionable = sum(k in xl for k in ["recommend", "should", "consider", "avoid", "use", "suggest"])
        fluff = sum(k in xl for k in ["it seems", "overall", "in general", "various"])

        s = 1.0 + 1.4 * risk + 0.7 * actionable + 0.5 * has_number - 1.0 * fluff
        s -= max(0, (len(x) - 170) / 200)  # penalize long sentences
        return s

    scored = [(i, score(i)) for i in insights]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:top_k]]


def build_report_context(
    eda: dict,
    eda_tables: dict,
    cleaning_stats: dict,
    cleaning_text: list,
    feature_report: list,
    target_column: str | None = None
):
    """
    Output: context dict used by:
    - UI renderer
    - PDF renderer
    - LLM writer sections
    """

    shape = eda.get("shape", (None, None))
    n_rows, n_cols = shape[0], shape[1]

    missing_table = eda_tables.get("missing_table")
    missing_top = []
    if isinstance(missing_table, pd.DataFrame) and not missing_table.empty:
        tmp = missing_table.sort_values("missing_%", ascending=False).head(5)
        missing_top = [
            f"{idx}: {row['missing_%']}%"
            for idx, row in tmp.iterrows()
        ]

    top_corr = eda.get("top_correlations", [])
    top_corr = top_corr[:5] if isinstance(top_corr, list) else []

    # Cleaning summary lines (already human readable)
    clean_lines = dedupe_sentences(_to_list(cleaning_text))

    # Feature engineering report
    feat_lines = dedupe_sentences(_to_list(feature_report))

    # EDA highlights (high-signal, not raw tables)
    highlights = []
    if n_rows is not None and n_cols is not None:
        highlights.append(f"Dataset contains {n_rows} rows and {n_cols} columns.")
    if missing_top:
        highlights.append("Highest missing columns: " + ", ".join(missing_top))
    if top_corr:
        highlights.append("Top correlations: " + "; ".join(top_corr))

    highlights = dedupe_sentences(highlights)
    highlights = rank_insights(highlights, top_k=6)

    context = {
        "dataset_name": "Uploaded Dataset",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target_column": target_column,
        "cleaning_lines": clean_lines,
        "feature_lines": feat_lines,
        "eda_highlights": highlights,
        "missing_top": missing_top,
        "top_correlations": top_corr
    }

    return context


def build_llm_brief(context: dict):
    """
    A compact evidence-only brief for LLM.
    """
    bullets = []

    bullets += context.get("eda_highlights", [])
    bullets += ["Cleaning actions:"] + [f"- {x}" for x in context.get("cleaning_lines", [])[:8]]
    bullets += ["Feature engineering:"] + [f"- {x}" for x in context.get("feature_lines", [])[:8]]

    bullets = dedupe_sentences(bullets)
    bullets = bullets[:20]

    return "\n".join(bullets)
