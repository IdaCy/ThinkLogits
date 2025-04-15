# faithfulness_plot.py

import json
import numpy as np
import matplotlib.pyplot as plt

MODEL_COLORS = ["#4CAF50", "#FF6347"]  

FRAME_COLOR = "#000000"
MODEL_HATCHES = [None, '//']
BAR_WIDTH = 0.35
BAR_SPACING_FACTOR = 1.0

def load_faithfulness(path):
    """
    Loads the raw_faithfulness value from a faithfulness_results.json file.
    Returns a float in [0,1]
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data["raw_faithfulness"]  # e.g. 0.8 means 80%

def plot_faithfulness(
    hint_types,
    model_names,
    paths, 
    out_file=None,
    bar_colors=MODEL_COLORS,
    frame_color=FRAME_COLOR,
    bar_width=BAR_WIDTH,
    hatches=MODEL_HATCHES,
    spacing_factor=BAR_SPACING_FACTOR
):
    """
    Plots a grouped bar chart- 2 models * N hint_types

    Parameters:
    -----------
    hint_types : list of str  - ["none", "sycophancy", "induced_urgency"]
    model_names : list of str - ["DeepSeek-R1-Llama-8B", "DeepSeek-R1-Qwen-1.5B"]
    paths : dict of dict
        A nested dict [hint][model], each storing the path to a .json
        - ["DeepSeek-R1-Llama-8B"] = "path/to/faithfulness_results.json"
    out_file : str or None
    """

    n_hints = len(hint_types)
    n_models = len(model_names)

    # Allocate array to store the fraction for each model/hint
    fractions = np.zeros((n_models, n_hints), dtype=float)

    # Load raw_faithfulness from each file
    for j, hint in enumerate(hint_types):
        for i, model in enumerate(model_names):
            frac = load_faithfulness(paths[hint][model])
            fractions[i, j] = frac * 100.0  # convert to percentage

    x = np.arange(n_hints)
    #offsets = np.linspace(-bar_width/2, bar_width/2, n_models)
    offsets = np.linspace(-bar_width/2, bar_width/2, n_models) * spacing_factor

    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(n_models):
        # If on matplotlib 3.7+, you can use hatchcolor=hatch_color
        # If on older versions, omit hatchcolor to avoid errors
        try:
            bar_container = ax.bar(
                x + offsets[i],
                fractions[i],
                width=bar_width / n_models * n_models,
                color=bar_colors[i],
                edgecolor=frame_color,
                hatch=hatches[i],
                label=model_names[i],
                zorder=3,
                #hatchcolor=hatch_color
            )
        except TypeError:
            bar_container = ax.bar(
                x + offsets[i],
                fractions[i],
                width=bar_width / n_models * n_models,  
                color=bar_colors[i],
                edgecolor=frame_color,
                hatch=hatches[i],
                label=model_names[i],
                zorder=3
            )

        # Add the numeric % labels on top of each bar
        for j, val in enumerate(fractions[i]):
            ax.text(
                x[j] + offsets[i], 
                val + 1.5, 
                f"{int(round(val))}%",
                ha='center', va='bottom', fontsize=9
            )

    # Format y-axis from 0% to 100%
    ax.set_ylim([0, 100])
    ax.set_ylabel("Fraction of examples with faithful CoT")
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    
    # x-axis ticks and labels
    ax.set_xticks(x)
    # Capitalize or nicely format your hint types for display
    display_labels = [h.replace("_", " ").title() for h in hint_types]
    ax.set_xticklabels(display_labels)
    ax.set_xlabel("Hints That The Model Might Use Without Verbalizing Them")

    # Add horizontal grid lines (behind the bars) for style
    ax.grid(True, axis='y', color='gray', linestyle='--', alpha=0.7, zorder=0)

    # Put legend in upper right
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    plt.show()
