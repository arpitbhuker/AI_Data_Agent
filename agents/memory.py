import json
import numpy as np
import pandas as pd
import hashlib
from pathlib import Path

MEMORY_PATH = Path("memory/agent_memory.json")


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """
    Create a stable fingerprint for a dataset
    based on structure + light content sampling.
    """
    hasher = hashlib.sha256()

    # Columns and dtypes
    schema_repr = "|".join(
        f"{col}:{str(dtype)}" for col, dtype in zip(df.columns, df.dtypes)
    )
    hasher.update(schema_repr.encode())

    # Shape
    hasher.update(str(df.shape).encode())

    # Sample values (first 5 rows, safe cast)
    sample = df.head(5).astype(str).values.flatten()
    hasher.update("".join(sample).encode())

    return hasher.hexdigest()


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_memory(memory: dict):
    safe_memory = _make_json_safe(memory)
    MEMORY_PATH.write_text(json.dumps(safe_memory, indent=2))


def load_memory():
    if MEMORY_PATH.exists():
        return json.loads(MEMORY_PATH.read_text())
    return {}
