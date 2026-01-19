import json
from pathlib import Path

def save_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2))
