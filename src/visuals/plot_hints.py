import json
import matplotlib.pyplot as plt
import numpy as np

def parse_switch_analysis(json_paths):
    """
    Given a list of file paths to switch_analysis .json files,
    reads them and returns:
      - switch_rate (fraction of questions that switched)
      - intended_rate (among switched, fraction that switched to intended hint)
      - correct_rate (fraction of questions that ended correct)
    """
    all_data = []
    for path in json_paths:
        with open(path, 'r') as f:
            all_data.extend(json.load(f))
            
    total = len(all_data)
    if total == 0:
        return 0.0, 0.0, 0.0
    
    switched = sum(1 for d in all_data if d.get("switched"))
    to_intended = sum(1 for d in all_data if d.get("switched") and d.get("to_intended_hint"))
    correct = sum(1 for d in all_data if d.get("is_correct_option"))
    
    switch_rate = switched / total
    intended_rate = to_intended / switched if switched > 0 else 0.0
    correct_rate = correct / total
    
    return switch_rate, intended_rate, correct_rate


def gather_switch_data(data_dict):
    """
    data_dict must be in the format:
      {
        'ModelName1': {
           'hint_type_1': 'path/to/switch_analysis_with_500.json',
           'hint_type_2': 'path/to/switch_analysis_with_500.json',
           ...
        },
        'ModelName2': {
           'hint_type_1': 'path/to/switch_analysis_with_500.json',
           ...
        },
        ...
      }
    
    This returns a nested dictionary:
      {
        'ModelName1': {
           'hint_type_1': (switch_rate, intended_rate, correct_rate),
           'hint_type_2': ...
        },
        'ModelName2': ...
      }
    """
    results = {}
    for model_name, hint_paths in data_dict.items():
        model_results = {}
        for hint_type, path_or_paths in hint_paths.items():
            # Accept either a single path or list of paths
            if isinstance(path_or_paths, str):
                path_or_paths = [path_or_paths]
            s_rate, i_rate, c_rate = parse_switch_analysis(path_or_paths)
            model_results[hint_type] = (s_rate, i_rate, c_rate)
        results[model_name] = model_results
    return results


def plot_switch_data(results, 
                     fig_title="Switch Analysis Results", 
                     figsize=(12, 5), 
                     bar_width=0.2):
    """
    Given the nested dictionary from gather_switch_data,
    creates a bar chart with three grouped bars per hint-type:
      - Switch Rate
      - Switch-to-Intended Rate (among switched)
      - Correct Rate
    
    Each group is subdivided by model.
    """
    # Collect all hint types (union across all models)
    # and ensure a stable ordering:
    hint_types = sorted({
        ht for model_res in results.values() for ht in model_res.keys()
    })

    # We will have 3 metrics:
    metric_names = ["Switch Rate", "To Intended Rate", "Correct Rate"]
    
    # Prepare figure
    fig, ax = plt.subplots(1, len(metric_names), figsize=figsize, sharey=False)
    if len(metric_names) == 1:
        # Just to handle the edge case if someone changed the code to a single plot
        ax = [ax]
    
    # x positions for the groups
    x = np.arange(len(hint_types))
    num_models = len(results.keys())
    
    # For each metric subplot
    for i, metric_name in enumerate(metric_names):
        subplot = ax[i]
        
        for m_idx, (model_name, model_res) in enumerate(results.items()):
            # Collect the metric across each hint_type in the correct order
            y_vals = []
            for ht in hint_types:
                # (switch_rate, intended_rate, correct_rate)
                rates_tuple = model_res.get(ht, (0.0, 0.0, 0.0))
                y_vals.append(rates_tuple[i])  # pick the correct metric index
                
            # offset each model's bar cluster
            offset = (m_idx - (num_models - 1)/2) * bar_width
            
            subplot.bar(
                x + offset, 
                y_vals,
                bar_width, 
                label=model_name
            )
        
        subplot.set_title(metric_name, fontsize=12)
        subplot.set_xticks(x)
        subplot.set_xticklabels(hint_types, rotation=30, ha='right')
        subplot.set_ylim([0, 1])    # since these are fractions
        
        if i == 0:
            subplot.set_ylabel("Fraction", fontsize=12)
        
        subplot.legend(fontsize=8)
    
    fig.suptitle(fig_title, fontsize=14)
    plt.tight_layout()
    plt.show()
