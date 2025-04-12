def parse_answer(text: str, marker: str = "So I'll finalize the answer as: ") -> str:
    """
    A helper to find the final letter in the model's full generation.
    """
    # find the marker, then read up to next whitespace or punctuation
    start_idx = text.find(marker)
    if start_idx == -1:
        return ""  # Not found
    start_idx += len(marker)
    # Try to read the next 5-10 characters
    snippet = text[start_idx:start_idx+10].strip()
    # e.g., if the model wrote "A\nSure blah" - take the first non-whitespace token
    for char in snippet:
        if char.upper() in ["A", "B", "C", "D"]:
            return char.upper()
    return ""
