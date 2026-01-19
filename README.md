# 🤖 AI Data Agent — Autonomous EDA 

AI Data Agent is an autonomous data analysis assistant that converts a raw CSV dataset into a full analyst-style EDA workflow: column profiling, cleaning decisions, EDA tables, smart visualizations, target-aware analysis, and a polished PDF report — along with a modeling-ready .ipynb export.

Instead of manually doing repetitive EDA steps, users can simply upload a dataset and instantly receive structured insights and deliverables.

- dataset profiling (column types, high-null, ID-like, constant columns)
- cleaning + feature engineering summaries
- smart chart selection (only meaningful plots)
- correlation analysis + top relationships
- professional AI-written EDA narrative report
- PDF report export (tables + charts + narrative)
- modeling handoff notebook export (`.ipynb`)

Built for **real-world analyst workflows**, where the goal is not to plot everything, but to generate the **right insights with reasoning**.

---

## 📌 Why This Project?

Most EDA tools either:
- generate excessive random charts (noise)
- show raw JSON outputs (not readable)
- fail to produce decision-ready reports
- require manual notebook coding every time

AI Data Agent solves that by behaving like a **junior analyst working autonomously**, producing structured insights & deliverables.

---

## ✨ Key Features

## 1) Column Profiling (Dataset Brain)
Before doing anything else, the system profiles the dataset to identify:
- numeric / categorical / datetime-like columns
- constant columns (no value for modeling)
- high-null columns (quality issues)
- ID-like columns (unique ratio too high → usually not useful)

This helps the pipeline decide:
- what to plot
- what to drop
- what needs cleaning attention

## 2) Cleaning Pipeline (Rule-Based Analyst Behavior)
Cleaning includes:
- duplicate removal
- missing value imputation
  - categorical → mode
  - numeric → median
- cleaning report generation with reasoning

## 3) EDA Output as Tables (Not JSON)
Instead of dumping raw dictionaries, EDA is shown as clean tables:
- dataset overview
- missing values table (count + %)
- column dtypes + unique values
- numeric summary (describe + missing)
- correlation matrix
- top correlations table

## 4) Smart Visualization (Minimal but Valuable)
The visualization engine produces only **high-signal plots**:
- top numeric distributions (variance-based)
- correlation heatmap (limited columns → readable)
- strongest numeric relationship scatter plot
- categorical → numeric analysis
  - uses boxplot only when outliers exist
  - groups rare categories into “Other”
  - auto-switches orientation for long labels

## 5) AI Narrative Report Writer (Professional Quality)
The LLM is not used for random insight generation.

It is used like a **consultant-style report writer**, producing:

- Executive Summary
- Introduction
- Data Overview
- Data Cleaning
- Summary Statistics
- Missing Values Analysis
- Univariate / Bivariate Findings
- Feature Engineering rationale
- Outliers & Correlation analysis
- Conclusions + Recommendations
- Next Steps

✅ If evidence is missing for a section, it is removed automatically  
❌ No filler text like “I removed some sections…”

## 6) Export Deliverables
### 📄 PDF Report Export
Includes:
- cleaning summary
- assumptions
- EDA tables
- charts
- AI narrative report (structured)

### 📓 Notebook Export (`.ipynb`)
Exports an EDA→modeling notebook styled like Kaggle notebooks, containing:
- cleaning code
- feature engineering code
- EDA steps
- correlation analysis
- target-aware EDA (if target exists)
- modeling starter template

---

## 🏗️ Project Architecture
```bash
AI_DATA_AGENT/
│
├── app.py
│
├── agents/
│   ├── ingestion.py
│   ├── profiling.py
│   ├── cleaning.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── visualization.py
│   ├── assumptions.py
│   ├── feature_importance.py
│   ├── narrative_builder.py
│   ├── report_schema.py
│   ├── llm_narrator.py
│   ├── report.py
│   └── memory.py
│
├── utils/
│   ├── config.py
│   └── notebook_exporter.py
│
├── reports/
│   ├── EDA_Report.pdf
│   └── charts/
│
└── notebooks/
    └── eda_to_modeling.ipynb
```

---

## ⚙️ Tech Stack

- **Python**
- **Streamlit** — UI frontend
- **Pandas / NumPy** — data processing
- **Matplotlib / Seaborn** — charts
- **ReportLab** — PDF generation (tables + charts)
- **OpenRouter (Llama 3.1 8B Instruct)** — AI narrative generation
- **nbformat** — notebook generation
- **Rule-based intelligence** for profiling + chart decision

---

## 🚀 How It Works (Workflow)

1. Upload dataset (`.csv`)
2. Dataset profiling selects meaningful columns & risks
3. Cleaning is applied (duplicates + missing)
4. EDA tables are generated
5. Smart visualizations are selected
6. Narrative is written by LLM from compressed evidence
7. PDF report export with tables + charts + narrative
8. Notebook export for modeling handoff

---

## ✅ Installation

### 1) Clone the repository
```bash
git clone https://github.com/<your-username>/AI-Data-Agent.git
cd AI-Data-Agent
```

### 2) Create environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Add API Key
In utils/config.py:
```bash
OPENAI_API_KEY = "YOUR_OPENROUTER_KEY"
```

### 5) Run the app
```bash
streamlit run app.py
```

---

## 📌 Example Outputs

- ✅ Cleaned dataset preview
- ✅ Profiling summary (expandable)
- ✅ EDA tables (missing, dtypes, summary, correlations)
- ✅ Relevant charts only (not spam)
- ✅ Report-quality narrative
- ✅ PDF Export with full structure
- ✅ Kaggle-style exported notebook

---

## 🔮 Future Improvements

- RAG-based industry-specific EDA recommendations
(ex: finance, healthcare, retail templates)
- Add anomaly detection module
- Add drift monitoring for repeated datasets
- Add SHAP-based explainability after modeling
- Add multi-agent planner to make LLM truly agentic
- Add dataset schema-based feature engineering suggestions
- Add auto-detect target column (optional)

---

## ✅ Pros & Cons
### Pros

- ✅ End-to-end automation
- ✅ Smart plot selection (not generic chart spam)
- ✅ Deliverables: PDF + Notebook
- ✅ Professional narrative style
- ✅ Extensible agent architecture

### Cons

- ⚠️ Not a fully autonomous agent planner yet
- ⚠️ Narrative depends on LLM quality + prompt tuning
- ⚠️ Rule-based profiling may need tuning per dataset type

---

👤 Author

Arpit
- (AI & ML) | Data Science + Data Analysis
- 🔗 GitHub: https://github.com/arpitbhuker/
- 🔗 LinkedIn: https://www.linkedin.com/in/arpitbhuker/
