import json
import random
import os
import sys
import re # Added for slugifying

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

    # Write output files into subdirectories
    for hint_type in hint_types:
        hint_type_slug = slugify(hint_type)
        hint_output_dir = os.path.join(output_dir, hint_type_slug)
        os.makedirs(hint_output_dir, exist_ok=True) # Create subdirectory

        output_path = os.path.join(hint_output_dir, "hints.json") # Filename is always hints.json
        try:
            with open(output_path, 'w') as f:
                json.dump(output_data[hint_type], f, indent=2)
            print(f"Successfully wrote {len(output_data[hint_type])} hints to {output_path}")
        except IOError as e:
            print(f"Error writing file {output_path}: {e}")

if __name__ == "__main__":
    input_json_file = "data/gsm_mc_stage_formatted.json"
    # Output directory remains the base directory where subdirectories will be created
    output_directory = "data"
    format_data_with_hints(input_json_file, output_directory)
