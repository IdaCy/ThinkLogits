def parse_answer(text: str, marker: str = "So I'll finalize the answer as: ") -> str:
    """
    A helper to find the final letter in the model's full generation.
    We look after the marker for an uppercase letter A/B/C/D (the first we find).
    """
    start_idx = text.find(marker)
    if start_idx == -1:
        return ""  # Not found
    start_idx += len(marker)
    # reading a little more in case there's whitespace or punctuation
    snippet = text[start_idx:start_idx+20].strip()
    for char in snippet:
        if char.upper() in ["A", "B", "C", "D"]:
            return char.upper()
    return ""
