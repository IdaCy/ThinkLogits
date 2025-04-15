import json
import os
from typing import List, Dict, Optional
import torch
import logging
import time # For timing
import torch.nn.functional as F  # <-- i ADDED this for computing softmax probabilities

# Assuming utils are in the same directory or PYTHONPATH is set correctly
from src.utils.prompt_constructor import construct_prompt
from src.utils.model_handler import generate_completion, load_model_and_tokenizer

# Setup basic logging (optional, but good practice)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define known chat templates (can be expanded)
# Using the structure expected by tokenize_instructions: {instruction}
KNOWN_CHAT_TEMPLATES = {
    "llama": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "llama3": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "qwen": "User: {instruction}\nAssistant:" # A generic fallback
}

def get_chat_template(model_name: str) -> str:
    """Selects a chat template based on the model name."""
    model_name_lower = model_name.lower()
    if "llama" in model_name_lower or "llama" in model_name_lower:
        return KNOWN_CHAT_TEMPLATES["llama3"]
    elif "qwen" in model_name_lower: # Assuming mistral instruct models
        return KNOWN_CHAT_TEMPLATES["qwen"]
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

def save_results(results: List[Dict], dataset_name: str, hint_type: str, model_name:str, n_questions: int, suffix: str = ""):
    """Saves the results to a JSON file in the dataset/model/hint_type directory.

    Args:
        suffix: Optional string to differentiate files (e.g. '_with_probs').
    """
    # Construct path including model name directory and remove model name from filename
    # We'll add 'suffix' (e.g. '_with_probs') if provided
    filename = f"completions{suffix}_with_{str(n_questions)}.json"
    output_path = os.path.join("data", dataset_name, model_name, hint_type, filename)
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_path}: {e}")

########################################
# FOR PROBABILITIES !
########################################
def generate_completion_with_probs(
    model,
    tokenizer,
    device,
    prompts: List[Dict],
    chat_template: str,
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
) -> List[Dict]:
    """
    Similar to generate_completion, but also stores token-level probabilities
    for each of the four MCQ letters: A, B, C, and D (if they are single tokens).

    Returns:
        A list of dicts containing:
          - "question_id"
          - "prompt_text"
          - "completion_text"
          - "token_probs": [step-wise probability info for A/B/C/D]
    """
    # getting single-token IDs for A, B, C, D
    a_id = tokenizer.encode("A", add_special_tokens=False)
    b_id = tokenizer.encode("B", add_special_tokens=False)
    c_id = tokenizer.encode("C", add_special_tokens=False)
    d_id = tokenizer.encode("D", add_special_tokens=False)

    # !! need to go in check if any larger than 1
    if not (len(a_id) == len(b_id) == len(c_id) == len(d_id) == 1):
        logging.warning("A/B/C/D do not map to single tokens in this tokenizer; storing partial info anyway.")

    results = []

    # processing the prompts in mini-batches
    for start_idx in range(0, len(prompts), batch_size):
        batch = prompts[start_idx:start_idx+batch_size]
        # Prepare inputs
        batch_texts = [chat_template.format(instruction=item["prompt_text"]) for item in batch]
        # Tokenize
        encoded = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)

        # Generate with requested parameters, capturing scores
        with torch.no_grad():
            generation_output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )
        
        # generation_output.sequences: shape [batch_size, seq_len + new_tokens]
        # generation_output.scores: list of length (number of generated tokens),
        #   each: [batch_size, vocab_size] logits

        # eed to separate out the prefix length:
        prompt_lengths = [enc.sum().item() for enc in encoded['attention_mask']]  
        # ^ or more directly: prompt_lengths = (encoded['attention_mask'].sum(dim=1)).tolist()

        for i, item in enumerate(batch):
            seq_ids = generation_output.sequences[i]
            # The complete generated text (including prompt). We'll slice the new portion if desired:
            full_text = tokenizer.decode(seq_ids, skip_special_tokens=True)

            # Just store the portion after the prompt possible:
            # new_tokens = seq_ids[prompt_lengths[i]:]
            # completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            completion_text = full_text  # For simplicity: entire text

            # Collect step-wise probabilities for A, B, C, D
            token_probs = []
            # The number of newly generated tokens is len(generation_output.scores)
            # iterate over each step of generation:
            for step, step_logits in enumerate(generation_output.scores):
                # step_logits: [batch_size, vocab_size]
                # Softmax to get probabilities
                probs = F.softmax(step_logits[i], dim=-1)

                # If single-token, just retrieve prob. If multi-token, partial or skip
                a_prob = probs[a_id[0]].item() if len(a_id) == 1 else None
                b_prob = probs[b_id[0]].item() if len(b_id) == 1 else None
                c_prob = probs[c_id[0]].item() if len(c_id) == 1 else None
                d_prob = probs[d_id[0]].item() if len(d_id) == 1 else None

                # Identify which token got generated at this step:
                generated_token_id = generation_output.sequences[i][prompt_lengths[i] + step]
                generated_token_str = tokenizer.decode([generated_token_id], skip_special_tokens=True)

                token_probs.append({
                    "step_idx": step,
                    "generated_token": generated_token_str,
                    "A_prob": a_prob,
                    "B_prob": b_prob,
                    "C_prob": c_prob,
                    "D_prob": d_prob
                })

            results.append({
                "question_id": item["question_id"],
                "prompt_text": item["prompt_text"],
                "completion_text": completion_text,
                "token_probs": token_probs
            })

    return results


def save_results_with_probs(results: List[Dict], dataset_name: str, hint_type: str, model_name:str, n_questions: int):
    """
    Saves the results (including token_probs) to a JSON file with '_with_probs' in the filename.
    """
    save_results(results, dataset_name, hint_type, model_name, n_questions, suffix="_with_probs")


def generate_dataset_completions(
    model,
    tokenizer,
    model_name,
    device,
    dataset_name: str,
    hint_types: List[str], # e.g., ["none", "sycophancy"]
    batch_size: int = 8,
    max_new_tokens: Optional[int] = 512,
    n_questions: Optional[int] = None,
    store_probabilities: bool = False  # <-- ADDED
):
    """
    Loads a model, processes datasets for specified hint types (with and without hints),
    generates completions, and saves the results.

    Args:
        model_name: Name/path of the Hugging Face model.
        hint_types: List of hint type identifiers (used to find data files).
        batch_size: Batch size for generation.
        max_new_tokens: Maximum number of new tokens. None means generate until EOS.
        store_probabilities: If True, also compute and store token-level probabilities
                            for A/B/C/D at each generation step.
    """
    start_time = time.time()
    
    # --- 2. Select Chat Template --- 
    chat_template = get_chat_template(model_name)
    logging.info(f"Using chat template: {chat_template}")

    # --- 3. Process each hint type dataset --- 
    for hint_type in hint_types:
        logging.info(f"--- Processing dataset for hint type: {hint_type} ---")
        questions_data_path = os.path.join("data", dataset_name, "input_mcq_data.json")
        hints_data_path = os.path.join("data", dataset_name, f"hints_{hint_type}.json")
        
        data = load_data(questions_data_path)[:n_questions]
        hints = load_data(hints_data_path)[:n_questions]
        
        # Create a dictionary mapping question_id to hint_text
        hint_dict = {hint["question_id"]: hint["hint_text"] for hint in hints}
        
        # Add hint_text to each question entry
        for entry in data:
            entry["hint_text"] = hint_dict.get(entry["question_id"])

        logging.info(f"Generating completions for {hint_type}...")

        # Construct prompts
        prompts = []
        for entry in data:
            prompt_text = construct_prompt(entry)
            prompts.append({"question_id": entry["question_id"], "prompt_text": prompt_text})

        if store_probabilities:
            # Using the function that captures probabilities
            results = generate_completion_with_probs(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompts=prompts,
                chat_template=chat_template,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
            save_results_with_probs(results, dataset_name, hint_type, model_name, n_questions)
        else:
            # Use the original approach (no probabilities)
            results = generate_completion(
                model, tokenizer, device, prompts, 
                chat_template, batch_size, max_new_tokens
            )
            save_results(results, dataset_name, hint_type, model_name, n_questions)

    end_time = time.time()
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

# Example usage is typically handled in your notebook or another script.
if __name__ == "__main__":
    pass  # Placeholder
