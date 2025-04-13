import json
import os
from typing import List, Dict, Optional
import torch
import logging
import time # For timing

# Assuming utils are in the same directory or PYTHONPATH is set correctly
from utils.prompt_constructor import construct_prompt
from utils.model_handler import generate_completion, load_model_and_tokenizer

# Setup basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define known chat templates (can be expanded)
# Using the structure expected by tokenize_instructions: {instruction}
KNOWN_CHAT_TEMPLATES = {
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "mistral": "<s>[INST] {instruction} [/INST]",
    # Add other templates here based on model type identifier in model_name
    "default": "User: {instruction}\nAssistant:" # A generic fallback
}

def get_chat_template(model_name: str) -> str:
    """Selects a chat template based on the model name."""
    model_name_lower = model_name.lower()
    if "llama-3" in model_name_lower or "llama3" in model_name_lower:
        return KNOWN_CHAT_TEMPLATES["llama3"]
    elif "mistral" in model_name_lower: # Assuming mistral instruct models
        return KNOWN_CHAT_TEMPLATES["mistral"]
    # Add more specific checks if needed
    else:
        logging.warning(f"No specific chat template found for {model_name}. Using default.")
        return KNOWN_CHAT_TEMPLATES["default"]

def load_data(data_path: str) -> List[Dict]:
    """Loads JSON data from a file, handling potential errors."""
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return [] # Return empty list to indicate failure
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {data_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred loading {data_path}: {e}")
        return []

def save_results(results: List[Dict], hint_type: str):
    """Saves the results to a JSON file in the hint_type directory."""
    output_path = os.path.join("data", hint_type, "completions.json")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_path}: {e}")

def generate_dataset_completions(
    model_name: str,
    hint_types: List[str], # e.g., ["gsm_mc_urgency", "gsm_mc_psychophancy"]
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
):
    """
    Loads a model, processes datasets for specified hint types (with and without hints),
    generates completions, and saves the results.

    Args:
        model_name: Name/path of the Hugging Face model.
        hint_types: List of hint type identifiers (used to find data files like f'{data_base_dir}/{ht}.json').
        batch_size: Batch size for generation.
        max_new_tokens: Maximum number of new tokens. None means generate until EOS.
    """
    start_time = time.time()
    
    # --- 1. Load Model and Tokenizer --- 
    try:
        model, tokenizer, device = load_model_and_tokenizer(model_name)
    except RuntimeError as e:
        logging.error(f"Failed to initialize model: {e}. Aborting.")
        return
        
    # --- 2. Select Chat Template --- 
    chat_template = get_chat_template(model_name)
    logging.info(f"Using chat template: {chat_template}")

    # --- 3. Process each hint type dataset --- 
    for hint_type in hint_types:
        logging.info(f"--- Processing dataset for hint type: {hint_type} ---")
        questions_data_path = os.path.join("data", "gsm_mc_stage_formatted.json")
        hints_data_path = os.path.join("data", f"hint_type", "hints.json")
        
        # Load questions and hints data
        data = load_data(questions_data_path)
        hints = load_data(hints_data_path)
        
        # Create a dictionary mapping question_id to hint_text
        hint_dict = {hint["question_id"]: hint["hint_text"] for hint in hints}
        
        # Add hint_text to each question entry
        for entry in data:
            entry["hint_text"] = hint_dict.get(entry["question_id"])

            

        # --- Generate with hints --- 
        logging.info(f"Generating completions for {hint_type}...")
        prompts = []
        for entry in data:
            prompt_text = construct_prompt(entry) # Use original entry with hint
            prompts.append({"question_id": entry["question_id"], "prompt_text": prompt_text})
            
        
        results = generate_completion(
            model, tokenizer, device, prompts, 
            chat_template, batch_size, max_new_tokens
        )
        output_path_with_hints = os.path.join(output_base_dir, model_name.replace("/", "_"), f"{hint_type}_with_hints_completions.json")
        save_results(results, output_path_with_hints)
        

    # --- 4. Cleanup --- 
    logging.info("Cleaning up model and tokenizer...")
    del model
    del tokenizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        logging.info("CUDA cache cleared.")
        
    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

# Example of how to call the function (replace main() block)
if __name__ == "__main__":
    generate_dataset_completions(
        model_name="openai-community/gpt2",
        hint_types=["induced_urgency"], 
        batch_size=4,
        max_new_tokens=1024
    )
#     # Example with max_new_tokens=None
#     # generate_dataset_completions(
#     #     model_name="mistralai/Mistral-7B-Instruct-v0.2",
#     #     hint_types=["gsm_mc_urgency"],
#     #     batch_size=2,
#     #     max_new_tokens=None
#     # )


# Remove the old main() function and argparse code 