import json
from typing import List, Dict, Any, Optional

def evaluate_results(
    results_json_path: str,
    threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluates the multi-hint results from pipeline output (multi_hint_results.json).
    
    Args:
        results_json_path: path to the JSON file pipeline wrote, e.g. 'output/multi_hint_results.json'.
        threshold: the probability cutoff for seeing when p(correct_answer) first exceeds this value.
        verbose: if True, prints a summary to the notebook cell output.
    
    Returns:
        A dictionary containing:
          {
            "total_questions": int,
            "hint_type_correct_counts": { hint_type: int, ... },
            "hint_type_total_counts": { hint_type: int, ... },
            "overall_correct": int,
            "overall_total": int,
            "question_details": [
              {
                "index": ...,
                "task": ...,
                "correct_answer": ...,
                "completions": [
                  {
                    "hint_type": ...,
                    "final_answer": ...,
                    "is_correct": bool,
                    "first_threshold_step": Optional[int],
                    "first_mention_step": Optional[int]
                  },
                  ...
                ]
              },
              ...
            ]
          }
    """
    
    # Load the results from JSON
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Data format is a list of questions, each with "completions" array
    total_questions = len(data)
    hint_type_correct_counts = {}
    hint_type_total_counts = {}

    question_details = []

    for question_record in data:
        q_index = question_record["index"]
        task = question_record["task"]
        correct_answer = question_record["correct_answer"]
        completions = question_record["completions"]

        # summary for each completion
        comp_summaries = []
        for comp in completions:
            hint_type = comp["hint_type"]
            final_answer = comp["final_answer"]
            token_probs = comp["token_probabilities"]  # list of dicts
            generated_tokens = comp["generated_tokens"]

            # Tally how many times we saw this hint type
            hint_type_total_counts[hint_type] = hint_type_total_counts.get(hint_type, 0) + 1

            is_correct = (final_answer == correct_answer)
            if is_correct:
                hint_type_correct_counts[hint_type] = hint_type_correct_counts.get(hint_type, 0) + 1

            # Find the earliest step where p(correct_answer) > threshold
            first_threshold_step = None
            for t_info in token_probs:
                p_correct = t_info[f"p({correct_answer})"]
                if p_correct > threshold:
                    first_threshold_step = t_info["token_index"]
                    break

            # Find the earliest mention of the correct letter in the generated_tokens
            # (assuming it's its own token)
            first_mention_step = None
            for i, tok in enumerate(generated_tokens):
                if tok.strip().upper() == correct_answer:
                    first_mention_step = i
                    break

            comp_summaries.append({
                "hint_type": hint_type,
                "final_answer": final_answer,
                "is_correct": is_correct,
                "first_threshold_step": first_threshold_step,
                "first_mention_step": first_mention_step
            })

        question_details.append({
            "index": q_index,
            "task": task,
            "correct_answer": correct_answer,
            "completions": comp_summaries
        })

    # Summarise correctness by hint_type
    # overall correctness = sum of all correct / sum of all total
    overall_correct = sum(hint_type_correct_counts.values())
    overall_total = sum(hint_type_total_counts.values())

    # If verbose, print to the notebook cell
    if verbose:
        print("=== Evaluation Summary ===")
        print(f"Loaded {total_questions} questions from {results_json_path}")
        print(f"Probability threshold for correctness time-check: {threshold}\n")

        # Print hint-type correctness rates
        print("Hint-type correctness rates:")
        for htype in sorted(hint_type_total_counts.keys()):
            total_ht = hint_type_total_counts[htype]
            correct_ht = hint_type_correct_counts.get(htype, 0)
            pct = 100.0 * correct_ht / total_ht if total_ht > 0 else 0.0
            print(f"  {htype}: {correct_ht}/{total_ht} correct ({pct:.1f}%)")

        overall_pct = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0
        print(f"\nOverall correctness: {overall_correct}/{overall_total} = {overall_pct:.1f}%\n")

    # Return a dictionary with all relevant info
    return {
        "total_questions": total_questions,
        "hint_type_correct_counts": hint_type_correct_counts,
        "hint_type_total_counts": hint_type_total_counts,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "question_details": question_details
    }
