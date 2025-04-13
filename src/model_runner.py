import os
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
from typing import Tuple, List, Dict, Any

# so logs/ directory exists
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/run.log",
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

def load_model_and_tokenizer(model_name: str):
    """
    Loads the given model and tokenizer from Hugging Face on GPU.
    """
    logging.info(f"Loading model and tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True
    )
    model.eval().cuda()
    logging.info("Model loaded to GPU.")
    return tokenizer, model


def generate_with_token_probabilities_batched(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    batch_size: int = 8
) -> List[Tuple[str, List[str], List[Dict[str, Any]]]]:
    """
    Generate for multiple prompts in batches on the GPU.
    For each prompt, we return (full_text, generated_tokens, token_probabilities).
    token_probabilities is a list of dicts with "p(A)", "p(B)", ...
    """
    all_results = []

    # prompts into chunks of size batch_size
    for start_idx in range(0, len(prompts), batch_size):
        end_idx = start_idx + batch_size
        batch_prompts = prompts[start_idx:end_idx]
        logging.info(f"Processing prompts {start_idx} to {end_idx-1} out of {len(prompts)}...")

        # Tokenize all at once with padding
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        
        # generation_output:
        #  - sequences: [batch_size, input_length + max_new_tokens]
        #  - scores: list of length (# new tokens), each shape [batch_size, vocab_size]

        sequences = generation_output.sequences
        scores = generation_output.scores

        # iterate over each item in the batch
        for i in range(len(batch_prompts)):
            # original prompt length
            prompt_len_i = (attention_mask[i] == 1).sum().item()
            # Or: (input_ids[i] != tokenizer.pad_token_id).sum().item()

            seq_i = sequences[i].tolist()  # entire token IDs
            gen_ids_i = seq_i[prompt_len_i:]  # newly generated portion

            full_text = tokenizer.decode(seq_i, skip_special_tokens=True)

            # Gather token-by-token probabilities for "A/B/C/D" for this sequence
            # The i-th item in the batch has scores[t][i] for t in [0..(len(gen_ids_i)-1)]
            generated_tokens = []
            token_probabilities = []

            for t, score_distribution in enumerate(scores):
                # If t >= len(gen_ids_i), it means no more tokens for i-th prompt
                if t >= len(gen_ids_i):
                    break

                # shape: [batch_size, vocab_size]
                logits_i = score_distribution[i] 
                probs_i = torch.softmax(logits_i, dim=0)

                token_id = gen_ids_i[t]
                token_str = tokenizer.decode([token_id], skip_special_tokens=True)
                generated_tokens.append(token_str)

                pA = pB = pC = pD = 0.0
                for letter in ["A", "B", "C", "D"]:
                    letter_ids = tokenizer.encode(letter, add_special_tokens=False)
                    if len(letter_ids) == 1:
                        letter_id = letter_ids[0]
                        p_val = probs_i[letter_id].item()
                        if letter == "A":
                            pA = p_val
                        elif letter == "B":
                            pB = p_val
                        elif letter == "C":
                            pC = p_val
                        elif letter == "D":
                            pD = p_val

                token_probabilities.append({
                    "token_index": t,
                    "token_str": token_str,
                    "p(A)": pA,
                    "p(B)": pB,
                    "p(C)": pC,
                    "p(D)": pD
                })
            
            all_results.append((full_text, generated_tokens, token_probabilities))

    return all_results


def run_all_prompts_for_question(
    model,
    tokenizer,
    task: str,
    choices: Dict[str, str],
    correct_answer: str,
    parse_answer_func,
    max_new_tokens: int = 150,
    batch_size: int = 64
) -> List[Dict]:
    """
    1. Builds all prompts for this question (no hint, plus all hint types).
    2. Runs them in batches for efficiency.
    3. Parses final single-letter answer from the model's output.
    4. Returns a list of results, each with (prompt, full_text, final_answer, etc.)
    """

    from src.prompt_constructor import build_prompts_for_question
    prompts_info = build_prompts_for_question(task, choices, correct_answer)
    
    # run them all batched
    prompts = [p["prompt"] for p in prompts_info]
    logging.info(f"Running {len(prompts)} prompts for question: {task}")

    results_batched = generate_with_token_probabilities_batched(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size
    )

    # Now, match them up back to the prompts_info
    question_results = []
    for info, (full_text, gen_tokens, token_probs) in zip(prompts_info, results_batched):
        final_letter = parse_answer_func(full_text)
        
        # Print / log partial results
        logging.info(f"Hint type: {info['hint_type']}, Final Answer: {final_letter}")
        #print(f"[DEBUG] {info['hint_type']}: final_answer={final_letter}")  # optional debug print

        out = {
            "hint_type": info["hint_type"],
            "hint_text": info["hint_text"],
            "prompt": info["prompt"],
            "full_text": full_text,
            "generated_tokens": gen_tokens,
            "token_probabilities": token_probs,
            "final_answer": final_letter
        }
        question_results.append(out)

    return question_results
