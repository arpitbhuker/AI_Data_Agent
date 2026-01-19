import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTOML_MAX_MODELS = 10
RANDOM_SEED = 42
