# Comparing LLM Faithfulness with Correct vs. Incorrect Hints

This repository is a replication of Chen et al. (2022), "Reasoning Models Don't Always Say What They Think". The core method investigates Large Language Model (LLM) Chain-of-Thought (CoT) faithfulness by presenting an LLM with pairs of multiple-choice questions: one with the question alone, and one with the question plus a hint (which can point to either the correct or an incorrect answer). By comparing the model's CoT reasoning and final answers across these paired inputs, the goal is to assess faithfulness – specifically, analyzing when the model switches its answer due to the hint and whether its reasoning explicitly relies on the hint (verbalization).

- The paper investigates the faithfulness of Large Language Model (LLM) Chain-of-Thought (CoT) reasoning by comparing model answers to multiple-choice questions under different conditions: without hints, with hints pointing to the correct answer, and with hints pointing to incorrect answers. This code implements that experimental setup.

## Repository Structure

```
.
├── data/         # Datasets, hints, and experimental results
├── notebooks/    # Jupyter notebooks for running experiments
│   └── main_with_faithfulness.ipynb
├── src/          # Source code for pipeline and analysis
│   ├── data_processing/ 
│   ├── utils/
│   ├── main/
│   └── eval/
├── README.md
└── requirements.txt
```

## Pipeline Execution (via Notebook)

The typical workflow, orchestrated by `notebooks/main_with_faithfulness.ipynb`, involves:

1.  **Setup:** Load environment, define dataset name, hint types, model path.
2.  **(Optional) Hint Generation:** Run `src.data_processing.data_hint_formatting.format_data_with_hints` if hints haven't been generated yet.
3.  **Load Model:** Load the primary LLM and tokenizer using `src.main.pipeline.load_model_and_tokenizer`.
4.  **Generate Completions:** Run `src.main.pipeline.generate_dataset_completions` for the baseline ('none') and all specified hint types. This saves raw model outputs.
5.  **Verify Answers:** Run `src.eval.llm_verificator.run_verification` to extract the final MCQ answer from each completion using Gemini. Saves verification files.
6.  **Analyze Switches:** Run `src.eval.switch_check.run_switch_check` to compare baseline vs. hinted answers and determine switching behavior. Saves switch analysis files.
7.  **Verify Hint Usage:** Run `src.eval.llm_hint_verificator.run_hint_verification` on completions where the answer switched to the intended hint, checking if the hint was verbalized. Saves hint verification files.
8.  **(Optional) Calculate Faithfulness:** Run `src.eval.faithfulness_metric.compute_faithfulness_metric` using the generated completion, verification, and hint files.

## Usage

1.  **Install Dependencies:** `pip install -r requirements.txt`
2.  **Prepare Data:** Ensure input data is in `data/[dataset_name]/input_mcq_data.json` and corresponding hint files are present (or generate them using `src/data_processing/data_hint_formatting.py`).
3.  **Configure Notebook:** Open `notebooks/main_with_faithfulness.ipynb` and set parameters like `dataset_name`, `model_path`, etc.
4.  **Run Notebook:** Execute the cells sequentially.


