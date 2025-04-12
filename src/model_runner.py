import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig
)
from typing import Tuple, List, Dict, Any

def load_model_and_tokenizer(model_name: str):
    """
    Loads the given model and tokenizer from Hugging Face.
    for Qwen 1.5b or Llama 8B
    ->
      model_name = 'Qwen/Qwen-1.5b'
      model_name = '?'
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # might need: trust_remote_code=True

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    model = model.to('cuda')
    
    return tokenizer, model


def generate_with_token_probabilities(
    model, 
    tokenizer, 
    prompt: str,
    max_new_tokens: int = 128
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Generates a model response to prompt, capturing:
      - The final decoded text,
      - A list of tokens the model generated (step by step),
      - Probabilities (p(A), p(B), p(C), p(D)) at each generation step.

    Returns:
      full_text - entire generated text
      generated_tokens - each newly generated token (not including the prompt)
      token_probabilities:
        A list of dicts, one per new token:
          {
            "token_index": i,
            "token_str": "...",
            "p(A)": float,
            "p(B)": float,
            "p(C)": float,
            "p(D)": float
          }
    """
    # Tokenize the prompt
    # possibly: add_special_tokens=False
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    input_ids = input_ids.to(model.device)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        output_scores=True,         # return the logits
        return_dict_in_generate=True
    )

    # Generate
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config
    )

    # list of length max_new_tokens, each shape [batch_size, vocab_size]
    scores = generation_output.scores  # type: ignore
    # sequences: [batch_size, input_length + max_new_tokens]
    sequences = generation_output.sequences

    # decode entire sequence - but only to track the newly generated tokens beyond the original prompt length
    seq = sequences[0].tolist()
    prompt_len = input_ids.shape[1]
    generated_ids = seq[prompt_len:]  # newly generated token IDs

    full_text = tokenizer.decode(seq, skip_special_tokens=True)
    
    generated_tokens = []
    token_probabilities = []

    for i, score_distribution in enumerate(scores):
        # score_distribution: [batch_size, vocab_size]
        logits = score_distribution[0]
        probs = torch.softmax(logits, dim=0)

        token_id = generated_ids[i]
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)

        generated_tokens.append(token_str)

        # p tokens A B C D
        pA = 0.0
        pB = 0.0
        pC = 0.0
        pD = 0.0

        for letter in ["A", "B", "C", "D"]:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) == 1:  # exactly one token
                letter_id = encoded[0]
                p_val = probs[letter_id].item()
                if letter == "A":
                    pA = p_val
                elif letter == "B":
                    pB = p_val
                elif letter == "C":
                    pC = p_val
                elif letter == "D":
                    pD = p_val

        token_probabilities.append({
            "token_index": i,
            "token_str": token_str,
            "p(A)": pA,
            "p(B)": pB,
            "p(C)": pC,
            "p(D)": pD
        })
    
    return full_text, generated_tokens, token_probabilities
