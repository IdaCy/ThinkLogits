# Mechanistic Interpretability on Hinted vs. Unhinted Prompts

This repo compares how large language models (Qwen 1.5b & Llama 8B) respond to multiple-choice questions with or without a correct hint. We plan to measure the model’s internal probabilities for each MCQ option during chain-of-thought generation—examining if (and when) the model “locks in” on the correct answer, and how a hint might accelerate that certainty.


## Overview

building on Chen et al. (2022), “Reasoning Models Don’t Always Say What They Think”  

MCQ tasks to a model under two conditions:
  1. No Hint: The model only sees the question and the four answer choices (A, B, C, D)  
  2. Hinted: The model sees the question, the four options, and a hint that states the correct answer (e.g., “I think the answer is A, but what do you think?”)  

Insert a special line in the chain-of-thought:    
  “Sure! I think the correct answer is the MCQ option {}.”   

so we can observe exactly when the model’s probabilities for each option spike.

We collect token-by-token probabilities for each final MCQ letter (A, B, C, or D)   
- Does the model’s token distribution shift immediately after seeing the hint?  
- Does the chain-of-thought appear to weigh different options even if the model’s internal distribution is already confident?  

want to see how hints affect the model’s generation and when it decides on the correct answer  

## Repository Structure

```
├── data/
├── notebooks/
│   └── main.ipynb
├── src/
│   ├── data_reader.py
│   ├── prompt_constructor.py
│   ├── model_runner.py
│   └── pipeline.py
│ prompts, and saving results
├── .gitignore
├── README.md
└── requirements.txt
```


## Data Format

```
{
  "task": "2 + 2 = ?",
  "answer": "A",
  "A": "4",
  "B": "5",
  "C": "6",
  "D": "7"
}
```

## Usage

### 1. Prepare Data

- dataset in `data/<?>.json`

### 2. Run Notebook

1. Open `notebooks/main.ipynb`.  
2. Verify the `model_name` Can be:  
   ```
   model_name = "Qwen/Qwen-1.5b"
   # or
   model_name = "/path/to/llama-8b-checkpoint"
   ``` 
3. Run all cells. This will:  
   - Load data from `data/?.json`  
   - Build the prompt for each record (with or without hints).    
   - Generate token-by-token with the model, capturing probabilities for A/B/C/D.   
   - Saves a result to outputs/  


