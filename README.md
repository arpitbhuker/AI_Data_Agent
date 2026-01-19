# Autonomous AI Data Analysis Agent

This project implements a multi-agent AI system that autonomously:
- Cleans datasets
- Performs feature engineering
- Executes EDA
- Generates visualizations
- Trains models using AutoML
- Produces human-readable insights using LLMs
- Maintains memory across runs

## Architecture
Deterministic agents handle all data transformations.
LLMs are used strictly for narrative insight generation.

## How to Run
1. Install dependencies
2. Provide OpenAI API key
3. Run:
   streamlit run app.py
