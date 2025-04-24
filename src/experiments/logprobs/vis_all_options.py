# %% [markdown]
# Visualize All Option Trajectories for Individual Questions
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
# --- Helper Functions (from vis_individual) ---

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

# %% 
# --- Main Plotting Function ---

def plot_all_option_trajectories_for_qid(
    qid: str,
    q_data: dict,
    intervention_type: str,
    percentage_steps: list[int],
    model_name: str,
    dataset_name: str
):
    """Plots trajectories for all options (A,B,C,D) for a single question ID."""
    
    verified_answer = q_data.get('verified_answer')
    int_data = q_data.get(intervention_type)
    
    if not int_data or 'logprobs_sequence' not in int_data or not int_data['logprobs_sequence']:
        logging.warning(f"QID {qid}: Missing or empty 'logprobs_sequence' for intervention '{intervention_type}'. Skipping plot.")
        return
        
    logprobs_sequence = int_data['logprobs_sequence']
    num_steps = len(percentage_steps)
    
    if len(logprobs_sequence) != num_steps:
         logging.warning(f"QID {qid}: Sequence length mismatch (expected {num_steps}, got {len(logprobs_sequence)}). Skipping plot.")
         return
         
    # Assuming standard options A, B, C, D for simplicity - might need adjustment if options vary
    options = sorted(logprobs_sequence[0].get('logprobs', {}).keys())
    if not options or not all(len(opt) == 1 and opt.isupper() for opt in options):
        logging.warning(f"QID {qid}: Could not determine standard options (A,B,C,D...) from first step. Skipping plot.")
        return
        
    # Store trajectories for each option
    option_trajectories = {opt: np.full(num_steps, np.nan) for opt in options}

    # Calculate trajectories
    valid_steps = 0
    for i, step_data in enumerate(logprobs_sequence):
        logprobs = step_data.get('logprobs')
        if logprobs is None or set(logprobs.keys()) != set(options):
            logging.warning(f"QID {qid}, Step {percentage_steps[i]}%: Missing or mismatched logprobs. Setting step to NaN.")
            continue # Keep NaNs for this step for all options

        logprob_values = np.array([logprobs[opt] for opt in options])
        normalized_props = linear_normalize_logits(logprob_values)
        
        if normalized_props is None:
             logging.warning(f"QID {qid}, Step {percentage_steps[i]}%: Normalization failed. Setting step to NaN.")
             continue
             
        valid_steps += 1
        for opt, prop in zip(options, normalized_props):
            option_trajectories[opt][i] = prop
            
    if valid_steps == 0:
        logging.warning(f"QID {qid}: No valid steps found to plot.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(options)))
    
    for i, opt in enumerate(options):
        label = f"Option {opt}"
        marker = 'o' # Default marker
        linewidth = 1.5
        if opt == verified_answer:
            label += " (Verified)"
            linewidth = 2.5 # Make verified answer line thicker
            # marker = '*' # Optional: use a different marker
            
        plt.plot(percentage_steps, option_trajectories[opt], marker=marker, 
                 linestyle='-', color=colors[i], label=label, linewidth=linewidth)

    plt.title(f"QID {qid}: All Option Trajectories (Baseline / {intervention_type})\n{model_name} on {dataset_name}")
    plt.xlabel("Reasoning Step (%)")
    plt.ylabel("Normalized Logit Proportion")
    plt.xticks(percentage_steps)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show() # Display the plot

# %% 
# --- Script Execution Control ---

def run_baseline_all_option_plots(
    dataset_name: str,
    model_name: str,
    intervention_type: str,
    num_questions_to_plot: int = 3
):
    """Loads baseline data and plots all option trajectories for N random questions."""
    logging.info(f"--- Plotting All Baseline Options for Intervention: {intervention_type} ---")
    base_results_dir = os.path.join("src", "experiments", "logprobs", "results", dataset_name, model_name)
    baseline_file = os.path.join(base_results_dir, "baseline_logprobs.json")
    
    baseline_data = load_logprobs_data(baseline_file)
    if not baseline_data or 'results' not in baseline_data:
        logging.error(f"Could not load or parse baseline data from {baseline_file}")
        return
        
    percentage_steps = baseline_data.get("experiment_details", {}).get("percentage_steps", [])
    if not percentage_steps:
        logging.error("Could not determine percentage steps from baseline experiment details.")
        return
        
    # Get all valid question IDs for the specific intervention type
    valid_qids = []
    for qid, q_data in baseline_data['results'].items():
        if 'error' not in q_data and q_data.get(intervention_type) and q_data.get('verified_answer'):
            valid_qids.append(qid)
            
    if not valid_qids:
        logging.warning(f"No valid baseline questions found for intervention '{intervention_type}'.")
        return
        
    # Select random subset of questions
    num_to_select = min(num_questions_to_plot, len(valid_qids))
    selected_qids = random.sample(valid_qids, num_to_select)
    logging.info(f"Selected QIDs for baseline plots: {selected_qids}")
    
    # Generate plot for each selected QID
    for qid in selected_qids:
        q_data = baseline_data['results'][qid]
        plot_all_option_trajectories_for_qid(
            qid=qid,
            q_data=q_data,
            intervention_type=intervention_type,
            percentage_steps=percentage_steps,
            model_name=model_name,
            dataset_name=dataset_name
        )
        
# %% 
# --- Example Execution ---
# Configuration
DATASET = "mmlu"
MODEL = "DeepSeek-R1-Distill-Llama-8B"
INTERVENTION = "dots" # Choose one intervention type to plot
N_QUESTIONS = 10 # Number of questions to plot

run_baseline_all_option_plots(
    dataset_name=DATASET,
    model_name=MODEL,
    intervention_type=INTERVENTION,
    num_questions_to_plot=N_QUESTIONS
) 
# %%
