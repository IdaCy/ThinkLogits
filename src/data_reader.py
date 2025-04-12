import json
from typing import List, Dict, Any

def load_data(json_file_path: str) -> List[Dict[str, Any]]:
    """
    Reads a JSON file containing multiple MCQ tasks.
    Each record would look like:
      {
        "task": "...",
        "A": "...",
        "B": "...",
        "C": "...",
        "D": "...",
        "hint": "...",     # or None / null
        "hint_type": "correct_hint" or "none"
      }
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
