# import os
# import json
# import requests
# import streamlit as st
# import traceback
# from datetime import datetime
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import CrossEncoder

# # Constants
# INDEX_PATH = "embeddings/faiss_index"
# LOG_DIR = "logs"
# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3:8b"
# MAX_CONTEXT_LEN = 1200

# # Setup
# os.makedirs(LOG_DIR, exist_ok=True)

# @st.cache_resource
# def load_vectorstore():
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
#     return FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# @st.cache_resource
# def load_reranker():
#     return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# vectorstore = load_vectorstore()
# reranker = load_reranker()

# def rerank(query, docs, top_k=3):
#     pairs = [(query, doc.page_content) for doc in docs]
#     scores = reranker.predict(pairs)
#     ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
#     return [doc for _, doc in ranked_docs[:top_k]]

# def stream_ollama(prompt):
#     response = requests.post(
#         OLLAMA_URL,
#         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
#         stream=True
#     )
#     response.raise_for_status()
#     partial_answer = ""
#     for line in response.iter_lines():
#         print("Raw line:", line)
#         if line:
#             try:
#                 json_line = json.loads(line)
#                 token = json_line.get("response", "")
#                 yield token
#             except Exception as e:
#                 print("JSON parse error:", e)
#     return partial_answer

# # UI
# st.title("🔎 RAG Q&A System with Llama")

# query = st.text_input("Ask a question:", "")

# if st.button("Search"):
#     if not query.strip():
#         st.warning("Please enter a valid query.")
#     else:
#         with st.spinner("Searching..."):
#             try:
#                 docs = vectorstore.similarity_search(query, k=3)
#                 top_docs = rerank(query, docs)

#                 context = "\n\n".join([doc.page_content for doc in top_docs])[:MAX_CONTEXT_LEN]
#                 prompt = f"""Answer the following question based on the context below.

# Context:
# {context}

# Question:
# {query}

# Answer:"""

#                 print("Retrieved docs:", len(docs))
#                 print("Top reranked docs:", len(top_docs))
#                 print("Prompt:", prompt)

#                 # st.success("Answer generated below ⬇️")
#                 # st.markdown("### 💬 Answer:")

#                 # Streaming response (fixing st.write and partial display)
#                 full_response = ""
#                 response_placeholder = st.empty()
#                 for token in stream_ollama(prompt):
#                     full_response += token
#                     response_placeholder.markdown(f"### 💬 Answer:\n\n{full_response}")

#                 # Logging
#                 ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 log_data = {"timestamp": ts, "query": query, "context": context, "answer": full_response}
#                 with open(f"{LOG_DIR}/log_{ts}.json", "w", encoding="utf-8") as f:
#                     json.dump(log_data, f, indent=2, ensure_ascii=False)

#             except Exception as e:
#                 st.error("❌ An error occurred.")
#                 st.code(traceback.format_exc())  # Shows full traceback


import os
import json
import requests
import streamlit as st
import traceback
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Constants
INDEX_PATH = "../embeddings/faiss_index"
LOG_DIR = "logs"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
MAX_CONTEXT_LEN = 1200

# Setup
os.makedirs(LOG_DIR, exist_ok=True)
st.set_page_config(page_title="RAG with Llama", layout="wide")


@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    return FAISS.load_local(INDEX_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

vectorstore = load_vectorstore()
reranker = load_reranker()

def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked_docs[:top_k]]

def stream_ollama(prompt):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if line:
            try:
                json_line = json.loads(line)
                token = json_line.get("response", "")
                yield token
            except Exception as e:
                print("JSON parse error:", e)

def get_llm_score(prompt):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")
        
        try:
            # Find the first occurrence of a JSON object
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start == -1 or end == -1:
                return {
                    "correctness": 0.0,
                    "relevance": 0.0,
                    "similarity": 0.0,
                    "context_utilization": 0.0,
                    "faithfulness": 0.0,
                    "explanation": "No JSON found"
                }
            
            json_str = response_text[start:end+1]
            scores = json.loads(json_str)
            return {
                "correctness": float(scores.get("correctness", 0.0)),
                "relevance": float(scores.get("relevance", 0.0)),
                "similarity": float(scores.get("similarity", 0.0)),
                "context_utilization": float(scores.get("context_utilization", 0.0)),
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "explanation": scores.get("explanation", "No explanation provided")
            }
        except json.JSONDecodeError:
            return {
                "correctness": 0.0,
                "relevance": 0.0,
                "similarity": 0.0,
                "context_utilization": 0.0,
                "faithfulness": 0.0,
                "explanation": "Invalid JSON"
            }
    except Exception as e:
        print("❌ Error in get_llm_score:", e)
        return {
            "correctness": 0.0,
            "relevance": 0.0,
            "similarity": 0.0,
            "context_utilization": 0.0,
            "faithfulness": 0.0,
            "explanation": f"Error: {str(e)}"
        }

# -------------------- Streamlit UI -------------------- #




st.title("🔎 RAG Q&A System with Llama")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 🧠 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1], start=1):
        with st.expander(f"Q{i}: {q}"):
            st.markdown(f"**A:** {a}")

# Query input
query = st.text_input("Ask a question:", "")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Searching..."):
            try:
                docs = vectorstore.similarity_search(query, k=3)
                top_docs = rerank(query, docs)

                context = "\n\n".join([doc.page_content for doc in top_docs])[:MAX_CONTEXT_LEN]
                prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""

                # Stream LLM response
                full_response = ""
                response_placeholder = st.empty()
                for token in stream_ollama(prompt):
                    full_response += token
                    response_placeholder.markdown(f"### 💬 Answer:\n\n{full_response}")

                # Add to chat history
                st.session_state.chat_history.append((query, full_response))

                # Log
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_data = {
                    "timestamp": ts,
                    "query": query,
                    "context": context,
                    "answer": full_response
                }
                with open(f"{LOG_DIR}/log_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(log_data, f, indent=2, ensure_ascii=False)

            except Exception as e:
                st.error("❌ An error occurred.")
                st.code(traceback.format_exc())
