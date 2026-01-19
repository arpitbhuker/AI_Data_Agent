import streamlit as st

# ------------------ CORE AGENTS ------------------
from agents.ingestion import load_data
from agents.cleaning import clean_data
from agents.eda import generate_eda, target_eda
from agents.visualization import auto_visualize
from agents.insights import generate_insights

# ------------------ ADVANCED AGENTS ------------------
from agents.feature_engineering import engineer_features
from agents.feature_importance import feature_importance
from agents.assumptions import eda_assumptions
from agents.explanations import explain_eda
from agents.llm_narrator import narrate_insights
from agents.report import generate_pdf
from agents.profiling import profile_dataset
from agents.narrative_builder import build_report_context



# ------------------ MEMORY ------------------
from agents.memory import (
    dataset_fingerprint,
    load_memory,
    save_memory
)

# ------------------ EXPORT ------------------
from utils.notebook_exporter import export_notebook


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Data Agent",
    layout="wide"
)

st.title("🤖 Autonomous AI Data Analysis Agent")
st.caption(
    "Upload a dataset → Automatic cleaning, EDA, visualization, explanations, and handoff."
)


# ------------------ FILE UPLOAD ------------------
file = st.file_uploader("📂 Upload CSV file", type=["csv"])


# ================== MAIN PIPELINE ==================
if file:

    # ---------- INGESTION ----------
    df = load_data(file)
    st.success("✅ Dataset loaded successfully")

    # ---------- MEMORY ----------
    fingerprint = dataset_fingerprint(df)
    memory = load_memory()

    if fingerprint in memory:
        st.info("🧠 This dataset was analyzed before. Historical context available.")
    else:
        st.info("🆕 New dataset detected. Starting fresh analysis.")

    # ---------- CLEANING ----------
    df_cleaned, cleaning_stats, cleaning_text = clean_data(df)

    st.subheader("🧹 Data Cleaning Summary")
    for line in cleaning_text:
        st.write("•", line)

    profile = profile_dataset(df)

    st.subheader("🧠 Column Profiling Summary")

    with st.expander("🔍 View full column profiling details", expanded=False):
        st.json(profile)


    # ---------- OPTIONAL TARGET ----------
    st.subheader("🎯 Target Column (Optional)")
    target_column = st.selectbox(
        "Select target column (if applicable)",
        ["None"] + list(df_cleaned.columns),
        key="target_column_select"
    )

    if target_column == "None":
        target_column = None

    # ---------- EDA ----------
    eda_report, eda_tables = generate_eda(df_cleaned)

    st.subheader("📊 Exploratory Data Analysis")

    # 1) Shape
    st.markdown("### ✅ Dataset Overview")
    st.write(f"Rows: **{eda_report['shape'][0]}** | Columns: **{eda_report['shape'][1]}**")

    # 2) Missing table
    st.markdown("### 🕳️ Missing Values")
    if not eda_tables["missing_table"].empty:
        st.dataframe(eda_tables["missing_table"], width="stretch")
    else:
        st.success("No missing values found ✅")

    # 3) Dtypes table
    st.markdown("### 🧾 Column Types & Unique Values")
    st.dataframe(eda_tables["dtypes_table"], width="stretch")

    # 4) Numeric summary
    if "numeric_summary_table" in eda_tables:
        st.markdown("### 📈 Numeric Summary")
        st.dataframe(eda_tables["numeric_summary_table"], width="stretch")

    # 5) Correlation matrix
    if "correlation_table" in eda_tables:
        st.markdown("### 🔗 Correlation Matrix")
        st.dataframe(eda_tables["correlation_table"], width="stretch")

    # 6) Top correlations
    if "top_correlations_table" in eda_tables:
        st.markdown("### ⭐ Top Correlations")
        st.dataframe(eda_tables["top_correlations_table"], width="stretch")

    with st.expander("🧩 View raw EDA JSON (optional)", expanded=False):
        st.json(eda_report)


    # ---------- ASSUMPTIONS ----------
    assumptions = eda_assumptions(df_cleaned, target_column)

    st.subheader("📌 EDA Assumptions")
    for a in assumptions:
        st.write("•", a)

    # ---------- TARGET-AWARE EDA ----------
    if target_column:
        st.subheader("🎯 Target-aware Insights")
        for insight in target_eda(df_cleaned, target_column):
            st.write("•", insight)

    # ---------- FEATURE ENGINEERING (POST-EDA) ----------
    df_features, feature_report = engineer_features(df_cleaned)

    if target_column:
        st.subheader("📈 Feature Importance (Pre-model)")
        for i in feature_importance(df_features, target_column):
            st.write("•", i)
    
    # ---------- REPORT CONTEXT (SSOT for UI + PDF + LLM) ----------
    report_context = build_report_context(
        eda=eda_report,
        eda_tables=eda_tables,
        cleaning_stats=cleaning_stats,
        cleaning_text=cleaning_text,
        feature_report=feature_report,
        target_column=target_column
    )


    # ---------- RULE-BASED INSIGHTS ----------
    insights = generate_insights(
        eda_report,
        cleaning_stats
    )

        # ---------- AI REPORT NARRATIVE ----------
    st.subheader("🧠 AI EDA Report Narrative")
    narrative = narrate_insights(
        report_context,  
        cleaning_stats,
        feature_report
    )
    st.markdown(narrative)


    # ---------- VISUALIZATION ----------
    st.subheader("📉 Visual Analysis")
    plots = auto_visualize(df_cleaned)

    for fig in plots:
        st.pyplot(fig)

    # ---------- DATA PREVIEW ----------
    st.subheader("🧾 Cleaned Data Preview")
    st.dataframe(df_cleaned.head(), width="stretch")

    with st.expander("🧪 View Engineered Features"):
        st.dataframe(df_features.head(), width="stretch")

    # ---------- MEMORY UPDATE ----------
    memory[fingerprint] = {
        "cleaning": cleaning_stats,
        "features": feature_report,
        "eda_summary": str(eda_report)
    }
    save_memory(memory)

    # ---------- PDF EXPORT ----------
    st.subheader("📄 Report Export")

    if st.button("Generate EDA PDF Report"):
        pdf_path = generate_pdf(
        insights=cleaning_text + feature_report,
        llm_text=narrative,
        assumptions=assumptions,
        charts=plots,
        eda_tables=eda_tables
    )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️ Download EDA Report",
                data=f,
                file_name="EDA_Report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

    # ---------- NOTEBOOK EXPORT ----------
    st.subheader("📓 Modeling Handoff")

    if st.button("Export EDA → Modeling Notebook"):
        nb_path = export_notebook(target_column)

        with open(nb_path, "rb") as f:
            st.download_button(
                "⬇️ Download Notebook",
                data=f,
                file_name="eda_to_modeling.ipynb",
                mime="application/x-ipynb+json",
                key="download_notebook"
            )

else:
    st.warning("⚠️ Please upload a CSV file to start the autonomous analysis.")
