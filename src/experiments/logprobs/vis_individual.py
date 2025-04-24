# %% [markdown]
# Individual Logit Trajectory Visualization

# %%
%cd ..
%cd ..
%cd ..


# %% 
# Imports and Setup
import os
import json
import logging
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% 
# --- Helper Functions ---

def load_logprobs_data(filepath: str) -> dict | None:
    """Loads logprobs JSON data from a file."""
    if not os.path.exists(filepath):
        logging.error(f"Data file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None

def linear_normalize_logits(logprob_values: np.ndarray) -> np.ndarray | None:
    """Performs linear normalization on logit values."""
    num_options = len(logprob_values)
    shifted_logits = logprob_values
    min_logit = np.min(logprob_values)
    if min_logit <= 0:
        shift_value = abs(min_logit) + 1e-9
        shifted_logits = logprob_values + shift_value

    logit_sum = np.sum(shifted_logits)

    if abs(logit_sum) < 1e-9:
        logging.warning(f"Logit sum near zero during normalization. Returning uniform.")
        return np.full(num_options, 1.0 / num_options)
    else:
        return shifted_logits / logit_sum

def get_individual_trajectories(
    q_data: dict,
    intervention_type: str,
    target_option_key: str,
    num_steps: int
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Extracts target and avg_other trajectories for a single question."""
    
    target_option = q_data.get(target_option_key)
    if target_option is None:
        logging.warning(f"Missing '{target_option_key}'.")
        return None, None, None

    int_data = q_data.get(intervention_type)
    if not int_data or 'logprobs_sequence' not in int_data or not int_data['logprobs_sequence']:
        logging.warning(f"Missing or empty 'logprobs_sequence' for intervention '{intervention_type}'.")
        return None, None, target_option
    
    logprobs_sequence = int_data['logprobs_sequence']

    # Ensure sequence length matches expected steps
    if len(logprobs_sequence) != num_steps:
         logging.warning(f"Sequence length mismatch (expected {num_steps}, got {len(logprobs_sequence)}). Returning None.")
         return None, None, target_option

    q_target_traj = np.full(num_steps, np.nan)
    q_other_traj = np.full(num_steps, np.nan)

    for i, step_data in enumerate(logprobs_sequence):
        logprobs = step_data.get('logprobs')
        if logprobs is None:
            logging.warning(f"Missing logprobs for step {i}. Setting step to NaN.")
            continue # Keep NaN for this step

        options = list(logprobs.keys())
        logprob_values = np.array([logprobs[opt] for opt in options])
        
        normalized_props = linear_normalize_logits(logprob_values)
        if normalized_props is None: # Should not happen with current logic but check
             logging.warning(f"Normalization failed for step {i}. Setting step to NaN.")
             continue
        
        prop_map = {opt: p for opt, p in zip(options, normalized_props)}

        if target_option not in prop_map:
            logging.warning(f"Target option '{target_option}' not found in normalized props for step {i}. Options: {options}. Setting step to NaN.")
            continue

        q_target_traj[i] = prop_map[target_option]
        
        # Find the proportion of the highest *other* option
        other_props = {opt: p for opt, p in prop_map.items() if opt != target_option}
        if not other_props:
            q_other_traj[i] = np.nan # No other options to compare against
        else:
            max_other_prop = max(other_props.values())
            q_other_traj[i] = max_other_prop

    return q_target_traj, q_other_traj, target_option


# %% 
# --- Main Plotting Function ---

def plot_trajectories_for_one_question(
    qid: str,
    steps: list[int],
    target_traj: np.ndarray,
    other_traj: np.ndarray,
    title_prefix: str,
    target_option: str,
    target_label: str,
    other_label: str = "Max Other Option"
):
    """Creates and displays a plot for a single question."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, target_traj, marker='o', linestyle='-', label=f"{target_label}: {target_option}")
    plt.plot(steps, other_traj, marker='x', linestyle='--', label=other_label)

    plt.title(f"{title_prefix} - QID: {qid}")
    plt.xlabel("Reasoning Step (%)")
    plt.ylabel("Normalized Logit Proportion")
    plt.xticks(steps)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show() # Display the plot
    plt.close() # Close the figure to free memory

def generate_plots_for_all_baseline_questions(
    dataset_name: str,
    model_name: str,
    intervention_type: str,
    n_questions: int = 5
):
    """Loads baseline data and plots individual trajectories for ALL valid questions."""
    logging.info(f"Generating baseline plots for Intervention: {intervention_type}")
    
    base_results_dir = os.path.join("src", "experiments", "logprobs", "results", dataset_name, model_name)
    input_file = os.path.join(base_results_dir, "baseline_logprobs.json")
    target_option_key = 'verified_answer'
    target_label_base = "Verified Answer"
    condition = 'baseline' # For logging/title

    logprobs_data = load_logprobs_data(input_file)
    if not logprobs_data or 'results' not in logprobs_data:
        logging.error(f"Could not load or parse data from {input_file}")
        return
        
    percentage_steps = logprobs_data.get("experiment_details", {}).get("percentage_steps", [])
    if not percentage_steps:
        logging.error("Could not determine percentage steps from experiment details.")
        return
    num_steps = len(percentage_steps)

    # Get all valid question IDs for the specific intervention type
    valid_qids = []
    for qid, q_data in logprobs_data['results'].items():
        # Check if qid data is valid for this intervention
        if ('error' not in q_data and 
            intervention_type in q_data and 
            q_data.get(target_option_key) and 
            'logprobs_sequence' in q_data[intervention_type] and 
            q_data[intervention_type]['logprobs_sequence']): 
             valid_qids.append(qid)
            
    if not valid_qids:
        logging.warning(f"No valid questions found for condition '{condition}' and intervention '{intervention_type}'.")
        return
        
    logging.info(f"Found {len(valid_qids)} valid questions for {condition} / {intervention_type}. Generating plots...")

    # Iterate through all valid questions
    # pick random n_questions
    selected_qids = random.sample(valid_qids, n_questions)
    for qid in selected_qids:
        q_data = logprobs_data['results'][qid]
        q_target_traj, q_other_traj, target_opt = get_individual_trajectories(
            q_data, intervention_type, target_option_key, num_steps
        )
        
        if q_target_traj is not None and q_other_traj is not None and target_opt is not None:
            # Check if trajectory calculation resulted in all NaNs (shouldn't happen often)
            if np.isnan(q_target_traj).all() or np.isnan(q_other_traj).all():
                logging.warning(f"Skipping plot for QID {qid} - trajectory calculation resulted in all NaNs.")
                continue

            plot_title_prefix = f"Baseline Trajectory ({intervention_type})\n{model_name} on {dataset_name}"
            plot_trajectories_for_one_question(
                qid=qid,
                steps=percentage_steps,
                target_traj=q_target_traj,
                other_traj=q_other_traj,
                title_prefix=plot_title_prefix,
                target_option=target_opt,
                target_label=target_label_base
            )
        else:
             logging.warning(f"Could not retrieve valid trajectory data for QID {qid}. Skipping plot.")

# %% 
# --- Example Execution ---
# Example usage: Plot baseline for all specified intervention types

# Configuration (replace with argparse if needed later)
DATASET = "mmlu"
MODEL = "DeepSeek-R1-Distill-Llama-8B"
# HINTS_TO_PLOT = ["sycophancy"] # Removed hints for now
INTERVENTIONS_TO_PLOT = ["dots", "dots_eot"]
N_QUESTIONS = 5 # Removed N

# Plot baseline for all valid questions
for intervention in INTERVENTIONS_TO_PLOT:
    generate_plots_for_all_baseline_questions(
        dataset_name=DATASET,
        model_name=MODEL,
        intervention_type=intervention,
        n_questions=N_QUESTIONS
    )
    
    # # Plot specified hint types - REMOVED FOR NOW
    # for hint in HINTS_TO_PLOT:
    #         for intervention in INTERVENTIONS_TO_PLOT:
    #             # Need a similar function for hints if required later
    #             pass 

# %%
