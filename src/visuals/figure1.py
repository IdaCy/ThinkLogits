#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import json
import os

# Define datasets, models, and hint types
datasets = ["MMLU"]
models = ["DeepSeek-R1-Llama-8B", "DeepSeek-R1-Qwen-14B"] # Use consistent naming for plotting
hint_types = ["Unethical Information", "Sycophancy", "Induced Urgency"] # Reordered to match target plot

# Map plot names to directory names
model_dir_names = {
    "MMLU": {
        "DeepSeek-R1-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B" # Map to the 14B dir for MMLU
    },
    "GSM8K": {
        "DeepSeek-R1-Llama-8B": "DeepSeek-R1-Distill-Llama-8B_old", # Map to the _old dir for GSM8K
        "DeepSeek-R1-Qwen-14B": "DeepSeek-R1-Distill-Qwen-14B"
    }
}
hint_type_dirs = {
    "Sycophancy": "sycophancy",
    "Unethical Information": "unethical_information",
    "Induced Urgency": "induced_urgency"
}

# Base path dictionary
base_paths = {
    "MMLU": "../../data/mmlu",
    "GSM8K": "../../data/gsm8k"
}

# Dictionary to store faithfulness scores {dataset: {model: {hint_type: score}}}
faithfulness_scores = {dataset: {model: {} for model in models} for dataset in datasets}

# Load data from JSON files
for dataset in datasets:
    base_path = base_paths[dataset]
    for model_display_name in models:
        model_dir_name = model_dir_names[dataset].get(model_display_name)
        if not model_dir_name:
            print(f"Warning: Directory name not found for {model_display_name} in {dataset}")
            continue # Skip if model directory mapping doesn't exist for this dataset

        for hint_display_name, hint_dir_name in hint_type_dirs.items():
            file_path = os.path.join(base_path, model_dir_name, hint_dir_name, "faithfulness_results.json")
            score = 0 # Default score is 0
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Use corrected_faithfulness score based on previous code, multiply by 100
                        score = data.get("corrected_faithfulness", 0) * 100
                except FileNotFoundError:
                    print(f"Warning: File not found at {file_path}")
                except json.JSONDecodeError:
                     print(f"Warning: Could not decode JSON for {dataset} - {model_display_name} - {hint_display_name}")
                except Exception as e:
                    print(f"An error occurred loading data for {dataset} - {model_display_name} - {hint_display_name}: {e}")
            else:
                print(f"Info: File does not exist, setting score to 0 for {file_path}")

            faithfulness_scores[dataset][model_display_name][hint_display_name] = score


print("Loaded Faithfulness Scores (%):")
print(json.dumps(faithfulness_scores, indent=2))


#%%
# --- Plotting ---
plt.style.use('seaborn-v0_8-ticks') # Use a style closer to the target

n_hints = len(hint_types)
n_datasets = len(datasets)
n_models = len(models)
n_bars_per_group = n_models * n_datasets

index = np.arange(n_hints)
bar_width = 0.18 # Adjusted bar width
group_spacing = 0.2 # Space between groups of bars for a given hint type

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors and hatches based on the target image
colors = {
    "DeepSeek-R1-Llama-8B": '#6aa84f', # Greenish
    "DeepSeek-R1-Qwen-14B": '#b4a7d6'  # Pinkish/Purplish
}
hatches = {
    "MMLU": '',
    "GSM8K": '//'
}
labels = {
    ("DeepSeek-R1-Llama-8B", "MMLU"): "DeepSeek-R1-Llama-8B (MMLU)",
    ("DeepSeek-R1-Llama-8B", "GSM8K"): "DeepSeek-R1-Llama-8B (GSM8K)",
    ("DeepSeek-R1-Qwen-14B", "MMLU"): "DeepSeek-R1-Qwen-14B (MMLU)",
    ("DeepSeek-R1-Qwen-14B", "GSM8K"): "DeepSeek-R1-Qwen-14B (GSM8K)"
}


# Plot bars for each model and dataset combination
bar_counter = 0
for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        scores = [faithfulness_scores[dataset][model].get(hint, 0) for hint in hint_types]

        # Calculate position for this bar within the group
        position_offset = (bar_counter - (n_bars_per_group - 1) / 2) * bar_width
        positions = index + position_offset

        label = labels.get((model, dataset), f"{model} ({dataset})") # Get specific label
        color = colors.get(model, '#000000') # Default to black if model not in colors dict
        hatch = hatches.get(dataset, '')

        bars = ax.bar(positions, scores, bar_width, label=label, color=color, hatch=hatch, edgecolor='black')

        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9) # Slightly smaller font

        bar_counter += 1


# Configure plot
ax.set_ylabel('Fraction of examples with faithful CoT') # Updated label
ax.set_xlabel('Hints That The Model Might Use Without Verbalizing Them') # Updated label
ax.set_xticks(index)
ax.set_xticklabels(hint_types)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mtick.PercentFormatter()) # Format y-axis as percentage

ax.legend(fontsize=9) # Adjust legend font size
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
ax.xaxis.grid(False) # Remove vertical grid lines
ax.spines['top'].set_visible(False) # Remove top spine
ax.spines['right'].set_visible(False) # Remove right spine


plt.tight_layout()
plt.show()

#%%
# --- Optional: Save the figure ---
# figure_save_path = "figure1_faithfulness_styled.png"
# fig.savefig(figure_save_path, dpi=300, bbox_inches='tight')
# print(f"Figure saved to {figure_save_path}") 
# %%
