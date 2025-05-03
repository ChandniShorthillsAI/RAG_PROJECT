import pandas as pd
from tqdm import tqdm
import json
import ollama
from concurrent.futures import ThreadPoolExecutor
import re

# Path to your CSV
csv_path = "../output_files/qa_with_context.csv"
output_path = "../output_files/custom_llm_rag_eval_gemma.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Define evaluation metrics
metrics = ["correctness", "relevance", "similarity", "context_utilization", "faithfulness"]

# Truncate helper (faster input, less cost)
def truncate(text, max_chars=300):
    return text[:max_chars] + "..." if len(text) > max_chars else text

# Prompt builder
def build_prompt(question, context, gt_answer, model_answer):
    return f"""
You are a helpful evaluator tasked with scoring how well the LLM-generated answer matches the ground truth, uses the context, and stays faithful. Give scores from 0 to 1.

Question: {question}

Retrieved Context: {context}

Ground Truth Answer: {gt_answer}

LLM Generated Answer: {model_answer}

Return a JSON object with the following fields:
- correctness
- relevance
- similarity
- context_utilization
- faithfulness
- explanation
"""

# Function to safely extract JSON from model response
def extract_json(response_text):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}")
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "similarity": 0.0,
            "context_utilization": 0.0,
            "faithfulness": 0.0,
            "explanation": response_text[:100] or "No output"
        }

# Get scores from gemma:2b via Ollama
def get_llm_score(prompt):
    try:
        response = ollama.chat(model="gemma:2b", messages=[
            {"role": "system", "content": "You are an evaluation assistant."},
            {"role": "user", "content": prompt}
        ])
        result_text = response["message"]["content"]
        result = extract_json(result_text)
        return result
    except Exception as e:
        print(f"⚠️ Evaluation failed: {e}")
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "similarity": 0.0,
            "context_utilization": 0.0,
            "faithfulness": 0.0,
            "explanation": "Evaluation failed"
        }

# Evaluate a single row
def evaluate_row(i, row):
    # Skip if already evaluated
    if not pd.isna(row.get("llm_correctness", None)) and row["llm_correctness"] > 0:
        return i, {}

    prompt = build_prompt(
        truncate(row["question"]),
        truncate(row["retrieved_context"]),
        truncate(row["answer"]),
        truncate(row["llm_answer"])
    )
    result = get_llm_score(prompt)

    row_result = {}
    for key in metrics:
        row_result[f"llm_{key}"] = result.get(key, 0.0)
    row_result["llm_explanation"] = result.get("explanation", "")
    return i, row_result

# Add new columns if not present
for key in metrics:
    col = f"llm_{key}"
    if col not in df.columns:
        df[col] = 0.0
if "llm_explanation" not in df.columns:
    df["llm_explanation"] = ""

# Multithreaded execution
print("🚀 Starting fast LLM evaluation with gemma:2b...")

with ThreadPoolExecutor(max_workers=6) as executor:  # adjust to your CPU
    futures = [executor.submit(evaluate_row, i, row) for i, row in df.iterrows()]

    for future in tqdm(futures):
        i, result = future.result()
        for key, val in result.items():
            df.at[i, key] = val

# Save the updated DataFrame
df.to_csv(output_path, index=False)
print(f"✅ Evaluation completed. Results saved to: {output_path}")
