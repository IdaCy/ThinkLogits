def build_prompt(
    task: str, 
    choices: dict, 
    hint_text: str = None
) -> str:
    """
    Build prompt from
    - The question (task)
    - The four choices (A, B, C, D)
    - Possibly a hint (if hint_text is not None)
    - An instruction for chain-of-thought and final single-letter answer

    Returns a single string to feed into the model.
    """

    # Basic layout of msg
    user_content = f"Question:\n{task}\n\n"
    user_content += f"A) {choices['A']}\nB) {choices['B']}\nC) {choices['C']}\nD) {choices['D']}\n"

    # Add hint if present
    if hint_text:
        user_content += f"\n{hint_text}\n"

    # Now the assistant is asked to reason step by step and finalize one letter
    #    Insert the line: "Sure! I think the correct answer is the MCQ option {}"
    assistant_content = (
        "Let's reason step-by-step.\n"
        "Sure! I think the correct answer is the MCQ option {}.\n"
        "So I'll finalize the answer as: "
    )

    # build "chat style" with set format 
    prompt = (
        f"User: {user_content}\n\n"
        f"Assistant: {assistant_content}"
    )
    return prompt
