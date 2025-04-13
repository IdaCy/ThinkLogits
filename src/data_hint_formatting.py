import json
import random
import os
import sys

# Add the parent directory to sys.path to allow importing from 'data'
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import hint templates
try:
    from data.hint_templates import hint_templates
except ImportError:
    print("Error: Could not import hint_templates from data.hint_templates.py")
    print("Please ensure the file exists and the script is run from the correct directory or the path is configured.")
    sys.exit(1)

def format_data_with_hints(input_file, output_dir):
    """
    Reads data from input_file, adds hints based on types, and writes to separate
    JSON files in output_dir.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read input data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return

    # Prepare data storage for each hint type
    hint_types = list(hint_templates.keys()) + ["None"]
    output_data = {hint_type: [] for hint_type in hint_types}

    # Process each item in the input data
    for item in data:
        correct_answer_option = item.get("correct")
        if not correct_answer_option:
            print(f"Warning: Skipping item due to missing or empty 'correct' field: {item}")
            continue

        for hint_type in hint_types:
            new_item = item.copy()
            if hint_type == "None":
                new_item['hint_type'] = None
                new_item['hint_text'] = None
            else:
                template = random.choice(hint_templates[hint_type])
                # Format the hint text by replacing {option} inside the brackets
                # with the correct answer key (e.g., "C")
                hint_text = template.replace("{option}", correct_answer_option)

                new_item['hint_type'] = hint_type
                new_item['hint_text'] = hint_text

            output_data[hint_type].append(new_item)

    # Write output files
    output_filenames = {
        "Sycophancy": "gsm_mc_sycophancy.json",
        "Unethical Information": "gsm_mc_unethical.json",
        "Induced Urgency": "gsm_mc_urgency.json",
        "None": "gsm_mc_none.json"
    }

    for hint_type, filename in output_filenames.items():
        output_path = os.path.join(output_dir, filename)
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data[hint_type], f, indent=2)
            print(f"Successfully wrote {len(output_data[hint_type])} items to {output_path}")
        except IOError as e:
            print(f"Error writing file {output_path}: {e}")

if __name__ == "__main__":
    input_json_file = "data/gsm_mc_stage_formatted.json"
    output_directory = "data"
    format_data_with_hints(input_json_file, output_directory)
