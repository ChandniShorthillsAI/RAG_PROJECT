import os
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from utils.ui_app import get_llm_score

# Get the absolute path to the CSV file
def get_csv_path():
    current_file = Path(__file__)
    return current_file.parent.parent / "output_files" / "qa_with_context.csv"

# Function to load the CSV data
def load_csv_data():
    csv_path = get_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}\nPlease ensure the file exists and try again.")
    return pd.read_csv(csv_path)

# Load the CSV data
try:
    df = load_csv_data()
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame()  # Empty DataFrame for testing

def truncate(text, max_chars=100):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def build_prompt(question, retrieved_context, answer, llm_answer):
    return f"""Question: {question}
Retrieved Context: {retrieved_context}
Correct Answer: {answer}
LLM Answer: {llm_answer}

Please evaluate the LLM's answer based on the following criteria:
1. Correctness: How accurate is the answer compared to the correct answer?
2. Relevance: How relevant is the answer to the question?
3. Similarity: How similar is the answer to the correct answer?
4. Context Utilization: How well does the answer use the retrieved context?
5. Faithfulness: How faithful is the answer to the retrieved context?

Please provide your evaluation in JSON format with scores between 0 and 1 for each criterion, and a brief explanation.
"""

def extract_json(response_text):
    try:
        # Find the first occurrence of a JSON object
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start == -1 or end == -1:
            return {"correctness": 0.0, "relevance": 0.0, "similarity": 0.0, 
                   "context_utilization": 0.0, "faithfulness": 0.0, "explanation": "No JSON found"}
        
        json_str = response_text[start:end+1]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"correctness": 0.0, "relevance": 0.0, "similarity": 0.0, 
               "context_utilization": 0.0, "faithfulness": 0.0, "explanation": "Invalid JSON"}

def evaluate_row(i, row):
    if 'llm_correctness' in row:
        return i, {}
        
    question = row['question']
    retrieved_context = row['retrieved_context']
    answer = row['answer']
    llm_answer = row['llm_answer']
    
    prompt = build_prompt(question, retrieved_context, answer, llm_answer)
    result = get_llm_score(prompt)
    
    return i, {
        'llm_correctness': result['correctness'],
        'llm_relevance': result['relevance'],
        'llm_similarity': result['similarity'],
        'llm_context_utilization': result['context_utilization'],
        'llm_faithfulness': result['faithfulness'],
        'llm_explanation': result['explanation']
    }

if __name__ == "__main__":
    # Apply evaluation to each row
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        _, result = evaluate_row(i, row)
        if result:
            results.append(result)
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "output_files" / "qa_score_summary.csv"
    results_df.to_csv(output_path, index=False)
