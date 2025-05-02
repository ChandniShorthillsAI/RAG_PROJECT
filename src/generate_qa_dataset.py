# import os
# import csv
# import json
# import requests
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from tqdm import tqdm

# # Constants
# INDEX_PATH = "embeddings/faiss_index"
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3:8b"
# OUTPUT_CSV = "qa_dataset.csv"

# # Load FAISS index
# embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
# vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# # Extract all chunks
# docs = vectorstore.docstore._dict.values()
# print(f"📄 Total chunks found: {len(docs)}")

# # Define prompt for Q&A generation
# # def build_prompt(text):
# #     return f"""
# # You are a helpful assistant. Given the following text, generate two factual question-answer pairs that directly relate to the content.

# # Text:
# # \"\"\"{text}\"\"\"

# # Format your response as JSON like this:
# # [
# #   {{
# #     "question": "Your question 1?",
# #     "answer": "Corresponding answer 1."
# #   }},
# #   {{
# #     "question": "Your question 2?",
# #     "answer": "Corresponding answer 2."
# #   }}
# # ]
# # """
# def build_prompt(text):
#     return f"""
# You are a factual assistant. Read the text below and generate exactly ONE factual question-answer pair from it.

# Text:
# \"\"\"{text}\"\"\"

# Respond **only** with valid JSON in this format:

#   {{
#     "question": "What is ...?",
#     "answer": "..."
#  }}

# """


# # def generate_qa(text):
# #     prompt = build_prompt(text)
# #     response = requests.post(
# #         OLLAMA_URL,
# #         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
# #     )
# #     response.raise_for_status()
# #     try:
# #         json_data = json.loads(response.json()['response'])
# #         return json_data
# #     except Exception as e:
# #         print("❌ Failed to parse JSON:", e)
# #         return []

# # # Generate QAs and write to CSV
# # with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
# #     writer = csv.DictWriter(f, fieldnames=["chunk_id", "question", "answer", "source"])
# #     writer.writeheader()

# #     for i, doc in enumerate(tqdm(docs, desc="🔍 Generating Q&A")):
# #         qa_pairs = generate_qa(doc.page_content[:1000])  # Limit long chunks
# #         for pair in qa_pairs:
# #             writer.writerow({
# #                 "chunk_id": i,
# #                 "question": pair.get("question", "").strip(),
# #                 "answer": pair.get("answer", "").strip(),
# #                 "source": doc.metadata.get("source", "N/A")
# #             })

# # print(f"\n✅ Dataset saved to `{OUTPUT_CSV}`")

# def generate_qa(text):
#     prompt = build_prompt(text)
#     try:
#         response = requests.post(
#             OLLAMA_URL,
#             json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
#         )
#         response.raise_for_status()

#         result = response.json()
#         raw_output = result.get("response", "").strip()

#         try:
#             qa_data = json.loads(raw_output)
#             return [qa_data]  # wrap in list for consistency
#         except json.JSONDecodeError:
#             print("🔍 Raw model output (truncated):", raw_output[:300])
#             print("❌ Failed to parse JSON, skipping chunk.")
#             return []

#     except Exception as e:
#         print("❌ Ollama call failed:", e)
#         return []






import os
import json
import csv
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

INDEX_PATH = "embeddings/faiss_index"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
OUTPUT_CSV = "generated_qa.csv"
MAX_CONTEXT_LEN = 1000

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# Get chunks
chunks = vectorstore.docstore._dict.values()
print(f"📄 Total chunks found: {len(chunks)}")

# Ensure CSV has headers
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer"])

# Prompt Template
def build_prompt(context):
    return f"""
Based on the context below, generate exactly one factual question and its correct answer. Format your response as JSON with keys "question" and "answer".

Context:
\"\"\"
{context}
\"\"\"

Your JSON response:
"""

# Stream response from Ollama
def get_qa_from_chunk(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode("utf-8"))
                    full_response += json_line.get("response", "")
                except Exception:
                    continue

        # Try extracting JSON object from full response
        start = full_response.find("{")
        end = full_response.rfind("}") + 1
        json_str = full_response[start:end]
        qa = json.loads(json_str)
        return qa.get("question", "").strip(), qa.get("answer", "").strip()

    except Exception as e:
        print(f"❌ Error processing chunk: {e}")
        return None, None

# Process each chunk
for chunk in tqdm(chunks, desc="🔍 Generating Q&A"):
    context = chunk.page_content[:MAX_CONTEXT_LEN]
    prompt = build_prompt(context)
    question, answer = get_qa_from_chunk(prompt)

    if question and answer:
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([question, answer])
