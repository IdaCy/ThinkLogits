import json
import random
import os
import sys
import re # Added for slugifying

# Add the parent directory to sys.path to allow importing from 'data'
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# print current working directory
print(os.getcwd())

# Import hint templates
try:
    # Adjusted path assuming hint_templates.py is in the parent 'data' directory
    from data.hint_templates import hint_templates
except ImportError:
    print("Error: Could not import hint_templates from data.hint_templates.py")
    print("Please ensure the file exists and the script is run from the correct directory or the path is configured.")
    sys.exit(1)


def slugify(text):
    """Convert string to lowercase, replace spaces with underscores."""
    return text.lower().replace(" ", "_")

def format_data_with_hints(input_file, output_dir):
    """
    Reads data from input_file, generates hints, and writes question_id and hint_text
    to separate JSON files in hint-type-specific subdirectories within output_dir.
    """
    # Ensure base output directory exists
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
    hint_types = list(hint_templates.keys())
    output_data = {hint_type: [] for hint_type in hint_types}

    # Process each item in the input data
    for item in data:
        correct_answer_option = item.get("correct")
        question_id = item.get("question_id") # Get question_id

        for hint_type in hint_types:
            hint_text = None
            if hint_type != "None":
                template = random.choice(hint_templates[hint_type])
                # Format the hint text by replacing {option} inside the brackets
                # with the correct answer key (e.g., "C")
                hint_text = template.replace("{option}", correct_answer_option)

            hint_entry = {
                "question_id": question_id,
                "hint_text": hint_text
            }
            output_data[hint_type].append(hint_entry)

    # Write output files directly into output_dir, skipping "None"
    for hint_type in hint_types:
        if hint_type == "None": # Skip writing the file for "None" hint type
            print(f"Skipping file generation for hint type: {hint_type}")
            continue

        # Construct output path directly in the output_dir
        # Using original hint_type string in filename for clarity
        output_path = os.path.join(output_dir, f"hints_{hint_type}.json")

        try:
            with open(output_path, 'w') as f:
                json.dump(output_data[hint_type], f, indent=2)
            print(f"Successfully wrote {len(output_data[hint_type])} hints to {output_path}")
        except IOError as e:
            print(f"Error writing file {output_path}: {e}")

if __name__ == "__main__":
    # Output directory becomes the dataset-specific directory (e.g., "data/gsm8k")
    # This would need to be passed dynamically, e.g., via argparse
    # Example for gsm8k:
    # dataset_name = "gsm8k"
    # input_json_file = os.path.join("data", dataset_name, "input_mcq_data.json")
    # output_directory = os.path.join("data", dataset_name)
    # format_data_with_hints(input_json_file, output_directory)
    # pass # Placeholder for actual execution logic
    format_data_with_hints("data/gsm8k/input_mcq_data.json", "data/gsm8k")
