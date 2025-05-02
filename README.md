
---

# 🧬 RAG-Based Chatbot on Human Evolution (Using Wikipedia + LLaMA 3)

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline that scrapes content from **Wikipedia on Human Evolution**, generates dense embeddings using **SentenceTransformers**, stores them in a **FAISS** index, and uses **LLaMA 3** (locally hosted via **Ollama**) to answer user questions based on the retrieved context. An evaluation pipeline measures the quality of generated answers using NLP metrics like **BERTScore** and **F1 score**.

---

## 📌 Components

### 1. 🧹 Web Scraping

- **Source**: Wikipedia page(s) related to **Human Evolution**.
- **Purpose**: Extract and clean textual data.
- **Output**: Combined text file saved under `data/`.

### 2. 🧠 Embeddings + FAISS Index

- **Embedding Model**: SentenceTransformers (`all-MiniLM-L6-v2` or similar).
- **Workflow**:
  - Chunk the scraped text using `RecursiveCharacterTextSplitter`.
  - Generate embeddings using SentenceTransformer.
  - Store the vectors in a FAISS index.

### 3. 🤖 RAG Pipeline (QnA)

- **Input**: Questions from `generated_qa.csv`.
- **Retrieval**: Top-k similar chunks via FAISS.
- **LLM**: `llama3` served locally using [Ollama](https://ollama.com/).
- **Pipeline Script**: `rag_pipeline.py`
- **Output**: `qa_with_context.csv` containing the question, ground truth, retrieved context, and LLM-generated answer.

### 4. 📊 Evaluation Pipeline

- **Input**: `qa_with_context.csv`
- **Implemented Metrics**:
  - **BERTScore**
  - **F1 Score**
- **Scripts**:
  - `evaluate_llm_answers.py`
- **Outputs**:
  - `qa_with_scores.csv`: Scores per question
  - `qa_score_summary.csv`: Mean scores across all questions

---

## 📁 Folder Structure

```
RAG_LLM/
├── data/
│   └── human_evolution_combined.txt         # Scraped Wikipedia data
│
├── embeddings/
│   ├── index.faiss                          # FAISS index
│   └── index.pkl                            # Mapping info
│
├── utils/                                   #contains scraper.py and ui_app.py file
│
├── output_files/
│   ├── generated_qa.csv                      # Input QA pairs
│   ├── qa_with_context.csv                   # QA + context + LLM answer
│   ├── qa_with_scores.csv                    # Evaluation per QA
│   └── qa_score_summary.csv                  # Overall evaluation summary
│
├── src/
│   ├── generate_qa_dataset.py                # Script to create QA pairs
│   ├── generate_llm_answers.py               # Script for RAG + answering
│   ├── evaluate_llm_answers.py               # BERTScore and F1
│   └── rag_pipeline.py                       # Core pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Start LLaMA 3 Locally
```bash
ollama run llama3
```

### 3. Generate Answers
```bash
python src/generate_llm_answers.py
```

### 4. Evaluate Answers
```bash
python src/evaluate_llm_answers.py
```

---

## ✅ Output Files

- `qa_with_context.csv`: Context + generated answers.
- `qa_with_scores.csv`: Scores (BERT, F1) per QA.
- `qa_score_summary.csv`: Aggregate metrics across all QAs.

---

## 🛠 Tech Stack

- **Python**
- **BeautifulSoup**, `requests` for web scraping
- **SentenceTransformers** for text embeddings
- **FAISS** for vector similarity search
- **Ollama** with **LLaMA 3** for local inference
- **NLTK**, **scikit-learn**, **evaluate** for QA scoring

---

