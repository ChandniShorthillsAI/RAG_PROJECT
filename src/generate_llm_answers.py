# import os
# import csv
# import json
# import time
# import requests
# import traceback
# from tqdm import tqdm
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# # Constants
# INDEX_PATH = "embeddings/faiss_index"
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3:8b"
# INPUT_CSV = "generated_qa.csv"
# OUTPUT_CSV = "qa_with_context.csv"
# MAX_CONTEXT_LEN = 1200

# # Load vectorstore
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# # Helper function to stream from Ollama
# def stream_ollama(prompt):
#     try:
#         response = requests.post(
#             OLLAMA_URL,
#             json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
#             stream=True
#         )
#         response.raise_for_status()
#         full_response = ""
#         for line in response.iter_lines():
#             if line:
#                 try:
#                     json_line = json.loads(line)
#                     token = json_line.get("response", "")
#                     full_response += token
#                 except json.JSONDecodeError:
#                     print("⚠️ JSON parsing error in Ollama stream.")
#         return full_response.strip()
#     except Exception as e:
#         print("❌ Ollama error:", e)
#         return ""

# # Resume mechanism
# def get_already_done_questions(output_path):
#     if not os.path.exists(output_path):
#         return set()
#     with open(output_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         return set(row["question"] for row in reader)

# # Main processing
# def generate_llm_answers():
#     already_done = get_already_done_questions(OUTPUT_CSV)

#     with open(INPUT_CSV, "r", encoding="utf-8") as infile, \
#          open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as outfile:

#         reader = csv.DictReader(infile)
#         fieldnames = ["question", "ground_truth", "retrieved_context", "llm_answer"]
#         writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#         if os.stat(OUTPUT_CSV).st_size == 0:
#             writer.writeheader()

#         for row in tqdm(reader, desc="🔄 Processing Questions"):
#             question = row["question"].strip()
#             # ground_truth = row["ground_truth"].strip()
#             ground_truth = row["answer"].strip()


#             if question in already_done:
#                 continue

#             try:
#                 docs = vectorstore.similarity_search(question, k=3)
#                 context = "\n\n".join([doc.page_content for doc in docs])[:MAX_CONTEXT_LEN]

#                 prompt = f"""Answer the following question based only on the context below.

# Context:
# {context}

# Question:
# {question}

# Answer:"""

#                 llm_answer = stream_ollama(prompt)

#                 writer.writerow({
#                     "question": question,
#                     "ground_truth": ground_truth,
#                     "retrieved_context": context,
#                     "llm_answer": llm_answer
#                 })

#                 outfile.flush()
#                 time.sleep(1)  # rate-limit safety

#             except Exception as e:
#                 print("❌ Error while processing:", question)
#                 print(traceback.format_exc())
#                 time.sleep(2)

# if __name__ == "__main__":
#     generate_llm_answers()

import os
import json
import time
import requests
import traceback
import pandas as pd
import csv
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
INDEX_PATH = "embeddings/faiss_index"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
INPUT_CSV = "generated_qa.csv"
OUTPUT_CSV = "qa_with_context.csv"
MAX_CONTEXT_LEN = 1200

# Load FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# Ollama streaming function
def stream_ollama(prompt):
    try:
        print("🔄 Sending prompt to Ollama...")
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True
        )
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line)
                    token = json_line.get("response", "")
                    full_response += token
                except json.JSONDecodeError:
                    print("⚠️ JSON parsing error in Ollama stream.")
        return full_response.strip()
    except Exception as e:
        print("❌ Ollama error:", e)
        return ""

# Resume: Read already processed questions
def get_done_questions():
    if os.path.exists(OUTPUT_CSV) and os.stat(OUTPUT_CSV).st_size > 0:
        df_done = pd.read_csv(OUTPUT_CSV)
        return set(df_done["question"].dropna().tolist())
    return set()

# Append data to CSV function
def append_to_csv(data):
    try:
        with open(OUTPUT_CSV, mode="a", newline="", encoding="utf-8") as outfile:
            fieldnames = ["question", "answer", "retrieved_context", "llm_answer"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            
            if os.stat(OUTPUT_CSV).st_size == 0:  # If the file is empty, write header
                writer.writeheader()
            
            writer.writerows(data)
            print(f"✅ Saved {len(data)} rows to {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ Error saving data to CSV: {e}")

# Main pipeline
def generate_llm_answers():
    input_df = pd.read_csv(INPUT_CSV)
    done_questions = get_done_questions()
    results = []

    # Process each question
    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="🔄 Processing Questions"):
        question = str(row["question"]).strip()
        ground_truth = str(row["answer"]).strip()

        if question in done_questions:
            continue

        try:
            # Retrieve context from FAISS vectorstore
            docs = vectorstore.similarity_search(question, k=6)
            context = "\n\n".join([doc.page_content for doc in docs])[:MAX_CONTEXT_LEN]

            # Prepare prompt for LLM
            prompt = f"""Answer the following question based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""

            # Get LLM-generated answer
            llm_answer = stream_ollama(prompt)

            # Append data to results list
            results.append({
                "question": question,
                "answer": ground_truth,
                "retrieved_context": context,
                "llm_answer": llm_answer
            })

            # Save data to CSV every 5 iterations (for safety)
            if len(results) >= 5:
                append_to_csv(results)
                results.clear()  # Clear results after saving

        except Exception as e:
            print("❌ Error while processing:", question)
            print(traceback.format_exc())
            time.sleep(2)

    # Final save if there are remaining results
    if results:
        append_to_csv(results)

# Run the script
if __name__ == "__main__":
    generate_llm_answers()








