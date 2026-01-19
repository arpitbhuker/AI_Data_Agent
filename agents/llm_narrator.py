from openai import OpenAI
from utils.config import OPENAI_API_KEY
from agents.report_schema import REPORT_SECTIONS
from agents.narrative_builder import build_llm_brief


client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


SYSTEM_PROMPT = """
You are a senior data analyst writing a professional Exploratory Data Analysis (EDA) report.

Rules:
- Use ONLY the evidence provided.
- Do NOT list raw dtypes, raw describe tables, or raw JSON.
- Keep tone professional, like a consulting report.
- Remove any section you cannot confidently write (do not leave empty).
- Avoid repetition.
- Focus on business + modeling impact.
- Avoid fluff like "overall" / "it seems" / "various".
- Never mention removed/missing sections.
- Never include meta commentary like "I removed" / "not included" / "lack of evidence".
- If a section has no evidence, silently omit it.
"""


REPORT_WRITER_PROMPT = """
Write an EDA report in this exact structure (use headings as shown).
If any section has no meaningful content based on the evidence, REMOVE that section completely.

Structure:
Exploratory Data Analysis (EDA) Report
Executive Summary
Introduction
Data Overview
Data Loading and Cleaning
Data Summary and Descriptive Statistics
Missing Values Analysis
Univariate Analysis
Bivariate Analysis
Feature Engineering
Outlier Detection
Correlation Analysis
Conclusions and Recommendations
Next Steps

Evidence:
{brief}

Output Requirements:
- Use markdown headings (## for major sections)
- Keep Executive Summary max 5 bullets
- Max ~600 words
"""


CRITIQUE_PROMPT = """
You are a strict reviewer.

Fix the draft to:
- remove duplicates
- remove vague filler (e.g. "overall", "it seems")
- improve clarity and professionalism
- ensure no empty sections exist
- keep headings and structure intact
- Do NOT repeat the report title if it already appears
- Output ONLY the final report
- Do NOT write: "Here is the revised draft", "Revised Draft", "Final Draft", etc

Draft:
{draft}
"""


def _remove_empty_sections(text: str) -> str:
    """
    Post-processing safety: remove headings that have no content.
    """
    lines = text.splitlines()
    cleaned = []
    buffer = []
    current_heading = None

    def flush_section():
        nonlocal buffer, current_heading
        if not buffer:
            return
        # if section contains only heading but no content -> drop
        content_lines = [l for l in buffer[1:] if l.strip()]
        if content_lines:
            cleaned.extend(buffer)
        buffer = []
        current_heading = None

    for line in lines:
        is_heading = line.strip().startswith("## ")
        if is_heading:
            flush_section()
            current_heading = line
            buffer = [line]
        else:
            if buffer:
                buffer.append(line)
            else:
                cleaned.append(line)

    flush_section()
    return "\n".join(cleaned).strip()


def narrate_insights(eda, cleaning, features):
    """
    Keep signature unchanged for your app.
    Here eda = context dict (not raw eda_report)
    cleaning = cleaning_stats
    features = feature_report
    """

    # The app will now pass report_context in "eda"
    context = eda
    brief = build_llm_brief(context)

    draft = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": REPORT_WRITER_PROMPT.format(brief=brief)}
        ],
        temperature=0.35
    ).choices[0].message.content

    refined = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": CRITIQUE_PROMPT.format(draft=draft)}
        ],
        temperature=0.2
    ).choices[0].message.content

    refined = _remove_empty_sections(refined)

    if not refined.startswith("#"):
        refined = f"# {REPORT_TITLE}\n\n" + refined

    return refined
