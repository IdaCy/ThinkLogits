# %% [markdown]
# Verbalization vs. Non-Verbalization Trajectory Visualization

# %% 
# Imports and Setup
import os
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# %% 
# --- Helper Functions (adapted from vis.py and vis_individual.py) ---

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

def process_verbalization_data(
    logprobs_data: dict,
    hint_type: str # Used for logging/context
) -> tuple[dict, list[int]]:
    """Processes loaded logprobs data, separating by verbalization status."""
    if not logprobs_data or 'results' not in logprobs_data:
        logging.warning("Invalid or empty logprobs data provided.")
        return {}, []

    # {intervention_type: {
    #     'verbalized': {'target': [traj1,...], 'other': [traj1,...]},
    #     'non_verbalized': {'target': [traj1,...], 'other': [traj1,...]}
    # }}
    trajectories = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    percentage_steps = logprobs_data.get("experiment_details", {}).get("percentage_steps", [])
    if not percentage_steps:
        logging.error("Could not determine percentage steps from experiment details.")
        return {}, []

    num_steps = len(percentage_steps)
    target_option_key = 'hint_option' # Always use hint_option for hinted data

    for qid, q_data in logprobs_data['results'].items():
        if 'error' in q_data:
            logging.warning(f"Skipping QID {qid} (Hint: {hint_type}) due to previous processing error: {q_data['error']}")
            continue
        
        status = q_data.get('status') # 'verbalized' or 'non_verbalized'
        target_option = q_data.get(target_option_key)

        if status not in ['verbalized', 'non_verbalized']:
            logging.warning(f"Skipping QID {qid} (Hint: {hint_type}): Invalid or missing status '{status}'.")
            continue
        if target_option is None:
            logging.warning(f"Skipping QID {qid} (Hint: {hint_type}): Missing '{target_option_key}'.")
            continue
            
        for int_type, int_data in q_data.items():
            # Skip metadata keys and intervention types with errors
            if int_type in ['status', 'depends_on_hint', 'quartiles', 'hint_option', 'is_correct_option', 'verified_answer', 'error']:
                 continue
            if 'logprobs_sequence' not in int_data or not int_data['logprobs_sequence']:
                logging.warning(f"Skipping QID {qid} (Hint: {hint_type}), Intervention '{int_type}': Missing or empty 'logprobs_sequence'.")
                continue

            logprobs_sequence = int_data['logprobs_sequence']
            if len(logprobs_sequence) != num_steps:
                 logging.warning(f"Skipping QID {qid} (Hint: {hint_type}), Intervention '{int_type}': Sequence length mismatch (expected {num_steps}, got {len(logprobs_sequence)}). Data might be incomplete.")
                 continue

            q_target_traj = np.full(num_steps, np.nan)
            q_other_traj = np.full(num_steps, np.nan)
            valid_sequence = True

            for i, step_data in enumerate(logprobs_sequence):
                logprobs = step_data.get('logprobs')
                if logprobs is None:
                    logging.warning(f"Missing logprobs for QID {qid}, Int '{int_type}', Step {i}. Setting step to NaN.")
                    q_target_traj[i], q_other_traj[i] = np.nan, np.nan
                    continue

                options = list(logprobs.keys())
                logprob_values = np.array([logprobs[opt] for opt in options])
                normalized_props = linear_normalize_logits(logprob_values)

                if normalized_props is None:
                    logging.warning(f"Normalization failed for QID {qid}, Int '{int_type}', Step {i}. Setting step to NaN.")
                    q_target_traj[i], q_other_traj[i] = np.nan, np.nan
                    continue
                
                prop_map = {opt: p for opt, p in zip(options, normalized_props)}

                if target_option not in prop_map:
                    logging.warning(f"Target option '{target_option}' not found for QID {qid}, Int '{int_type}', Step {i}. Setting step to NaN.")
                    q_target_traj[i], q_other_traj[i] = np.nan, np.nan
                    continue

                q_target_traj[i] = prop_map[target_option]
                other_props = {opt: p for opt, p in prop_map.items() if opt != target_option}
                q_other_traj[i] = np.nan if not other_props else max(other_props.values())

            # Only add if the whole sequence didn't have critical errors (like missing target)
            # Note: Individual step NaNs due to missing logprobs are handled by nanmean later
            # We might still have full NaN trajectories if *all* steps had missing logprobs
            trajectories[int_type][status]['target'].append(q_target_traj)
            trajectories[int_type][status]['other'].append(q_other_traj)

    # Average trajectories across questions for each group
    averaged_trajectories = defaultdict(dict)
    for int_type, status_data in trajectories.items():
        for status, traj_data in status_data.items():
            avg_target = np.nan # Default to NaN
            avg_other = np.nan
            count = len(traj_data.get('target', []))
            if count > 0:
                 # Convert lists of arrays to 2D numpy arrays for averaging
                target_array = np.array(traj_data['target'])
                other_array = np.array(traj_data['other'])
                avg_target = np.nanmean(target_array, axis=0)
                avg_other = np.nanmean(other_array, axis=0)
                
                if np.isnan(avg_target).all() or np.isnan(avg_other).all():
                    logging.warning(f"All NaN trajectory after averaging for {int_type}/{status}. Original count: {count}")
            else: 
                 logging.warning(f"No valid trajectories found for {int_type}/{status} to average.")

            averaged_trajectories[int_type][status] = {
                'target': avg_target,
                'other': avg_other,
                'count': count
            }

    return averaged_trajectories, percentage_steps


def plot_verbalization_comparison(
    steps: list[int],
    avg_data: dict, # Contains 'verbalized' and 'non_verbalized' keys
    title: str,
    save_path: str
):
    """Creates and saves a plot comparing verbalized vs. non-verbalized trajectories."""
    plt.figure(figsize=(12, 7))
    
    colors = {'verbalized': 'tab:blue', 'non_verbalized': 'tab:orange'}
    styles = {'target': ('o', '-'), 'other': ('x', '--')}

    for status, traj_groups in avg_data.items(): # status is 'verbalized' or 'non_verbalized'
        count = traj_groups.get('count', 0)
        if count == 0:
            logging.info(f"Skipping plotting for '{status}' in '{title}' - no data.")
            continue

        target_traj = traj_groups.get('target')
        other_traj = traj_groups.get('other')
        color = colors.get(status, 'black')
        
        # Plot target trajectory
        if target_traj is not None and not np.isnan(target_traj).all():
            marker, linestyle = styles['target']
            plt.plot(steps, target_traj, marker=marker, linestyle=linestyle, color=color, 
                     label=f"{status.capitalize()} (Hint Option, N={count})")
        
        # Plot max other trajectory
        if other_traj is not None and not np.isnan(other_traj).all():
            marker, linestyle = styles['other']
            plt.plot(steps, other_traj, marker=marker, linestyle=linestyle, color=color, alpha=0.8,
                     label=f"{status.capitalize()} (Max Other, N={count})")

    plt.title(title)
    plt.xlabel("Reasoning Step (%)")
    plt.ylabel("Average Normalized Logit Proportion")
    plt.xticks(steps)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Plot saved to {save_path}")

# %% 
# --- Main Function ---

def create_verbalization_visualizations(
    dataset_name: str,
    model_name: str,
    hint_types_to_analyze: list[str],
    intervention_types: list[str]
):
    """Loads logprobs results and generates verbalization comparison plots."""
    logging.info(f"Starting Verbalization comparison for Dataset: {dataset_name}, Model: {model_name}")
    base_results_dir = os.path.join("src", "experiments", "logprobs", "results", dataset_name, model_name)

    for hint_type in hint_types_to_analyze:
        logging.info(f"--- Processing Hint Type for Verbalization: {hint_type} --- ")
        input_file = os.path.join(base_results_dir, f"{hint_type}_logprobs.json")
        hint_data = load_logprobs_data(input_file)

        if not hint_data:
            logging.warning(f"Could not load data for hint type '{hint_type}'. Skipping.")
            continue
            
        avg_trajectories_by_verbalization, steps = process_verbalization_data(hint_data, hint_type)

        if not avg_trajectories_by_verbalization:
            logging.warning(f"No averaged trajectories could be calculated for hint type '{hint_type}'.")
            continue

        figure_dir = os.path.join(base_results_dir, hint_type, "figures_verbalization")

        for int_type, avg_data in avg_trajectories_by_verbalization.items():
            if int_type not in intervention_types: # Only plot requested intervention types
                 continue
                 
            plot_title = (f"Verbalization Comparison: {hint_type} ({int_type})\n"
                          f"{model_name} on {dataset_name}")
            save_path = os.path.join(figure_dir, f"{int_type}_verbalization_comparison.png")
            
            # Check if there is data for at least one status group
            has_verbalized_data = avg_data.get('verbalized', {}).get('count', 0) > 0
            has_non_verbalized_data = avg_data.get('non_verbalized', {}).get('count', 0) > 0
            
            if not has_verbalized_data and not has_non_verbalized_data:
                logging.warning(f"Skipping plot for {hint_type}/{int_type} - no data for either verbalized or non-verbalized.")
                continue
                
            plot_verbalization_comparison(
                steps=steps,
                avg_data=avg_data, 
                title=plot_title,
                save_path=save_path
            )

    logging.info("Verbalization comparison visualization finished.")



create_verbalization_visualizations(
        dataset_name="mmlu",
        model_name="DeepSeek-R1-Distill-Llama-8B",
        hint_types_to_analyze=["induced_urgency", "sycophancy", "unethical_information"],
        intervention_types=["dots", "dots_eot"]
    ) 

# # %% 
# # --- Script Execution ---

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate verbalization comparison trajectory visualizations.")
#     parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., mmlu)")
#     parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., DeepSeek-R1-Distill-Llama-8B)")
#     parser.add_argument("--hint_types", nargs='+', required=True, help="List of hint types to analyze (e.g., induced_urgency sycophancy)")
#     parser.add_argument("--intervention_types", nargs='+', required=True, help="List of intervention types used (e.g., dots dots_eot)")

#     args = parser.parse_args()

#     create_verbalization_visualizations(
#         dataset_name=args.dataset_name,
#         model_name=args.model_name,
#         hint_types_to_analyze=args.hint_types,
#         intervention_types=args.intervention_types
#     ) 