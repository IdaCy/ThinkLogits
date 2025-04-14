import json
from pathlib import Path
from typing import List
import os

def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ground_truth(file_path):
    """Loads ground truth data."""
    data = load_json(file_path)
    ground_truth = {item['question_id']: item['correct'] for item in data}
    return ground_truth

def load_verified_answers(file_path):
    """Loads verified answers from verification JSON files."""
    data = load_json(file_path)
    answers = {item['question_id']: item['verified_answer'] for item in data}
    return answers

def analyze_switches(base_answers, hint_answers, ground_truth):
    """Analyzes answer switches between base and hint types."""
    results = []
    # Find the intersection of question IDs present in both sets of answers
    common_q_ids = base_answers.keys() & hint_answers.keys()

    skipped_count = 0
    for q_id in common_q_ids:
        hint_answer = hint_answers[q_id]
        base_answer = base_answers[q_id]
        correct_answer = ground_truth[q_id]

        # Assuming answers are never None based on previous simplification request
        switched = hint_answer != base_answer
        to_hint = switched and (hint_answer == correct_answer)

        results.append({
            "question_id": q_id,
            "switched": switched,
            "to_hint": to_hint
        })
    return results

def calculate_accuracy(answers, ground_truth):
    """Calculates the accuracy of a set of answers against ground truth."""
    common_ids = answers.keys() & ground_truth.keys()
    if not common_ids:
        return 0, 0, 0.0  # Return zeros if no common questions

    correct_count = 0
    for q_id in common_ids:
        if answers[q_id] == ground_truth[q_id]:
            correct_count += 1

    total_comparable = len(common_ids)
    accuracy_percentage = (correct_count / total_comparable * 100)
    return correct_count, total_comparable, accuracy_percentage

def run_switch_check(dataset_name: str, hint_types: List[str], model_name: str, n_questions: int):
    """Runs the analysis and saves results per hint type."""
    DATA_DIR = Path("data") / dataset_name
    GROUND_TRUTH_FILE = DATA_DIR / "input_mcq_data.json"
    BASE_HINT_TYPE = "none"

    print("Loading ground truth...")
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)

    base_verification_file = DATA_DIR / BASE_HINT_TYPE / f"verification_{model_name}_with_{n_questions}.json"
    print(f"Loading base answers ({BASE_HINT_TYPE})...")
    base_answers = load_verified_answers(base_verification_file)

    # Calculate and print base accuracy using the new function
    base_correct_count, base_total_comparable, base_accuracy = calculate_accuracy(base_answers, ground_truth)
    print(f"Base ({BASE_HINT_TYPE}) Accuracy: {base_correct_count}/{base_total_comparable} ({base_accuracy:.2f}%)\n")

    all_results = {}

    for hint_type in hint_types:
        if hint_type == BASE_HINT_TYPE:
            continue

        hint_verification_file = DATA_DIR / hint_type / f"verification_{model_name}_with_{n_questions}.json"
        print(f"Processing hint type: {hint_type}...")
        hint_answers = load_verified_answers(hint_verification_file)

        # Calculate and print hint type accuracy
        hint_correct_count, hint_total_comparable, hint_accuracy = calculate_accuracy(hint_answers, ground_truth)
        print(f"  Accuracy: {hint_correct_count}/{hint_total_comparable} ({hint_accuracy:.2f}%)")

        results = analyze_switches(base_answers, hint_answers, ground_truth)
        all_results[hint_type] = results

        # Save results for the current hint type separately
        output_dir = DATA_DIR / hint_type
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        output_file_hint = output_dir / f"switch_analysis_{model_name}_with_{n_questions}.json"
        with open(output_file_hint, 'w') as f_hint:
            json.dump(results, f_hint, indent=2)
        print(f"Individual results for {hint_type} saved to {output_file_hint}")

    # --- Output ---
    print("\n--- Overall Results ---")
    for hint_type, results in all_results.items():
        print(f"\nHint Type: {hint_type}")
        total_comparable = len(results)
        total_switched = sum(1 for r in results if r['switched'])
        total_to_hint = sum(1 for r in results if r['to_hint'])
        switched_percentage = (total_switched / total_comparable * 100) if total_comparable else 0
        to_hint_percentage = (total_to_hint / total_comparable * 100) if total_comparable else 0
        print(f"  Total Entries: {total_comparable}")
        print(f"  Switched Answers: {total_switched} ({switched_percentage:.2f}%)")
        print(f"  Switched to Correct Answer: {total_to_hint} ({to_hint_percentage:.2f}%)")

# Example Usage (if run directly)
# if __name__ == "__main__":
#     run_switch_check(
#         dataset_name="gsm8k",
#         hint_types=["sycophancy", "induced_urgency", "unethical_information"], # Exclude 'none'
#         model_name="DeepSeek-R1-Distill-Llama-8B",
#         n_questions=150
#     )


