
---

# 🧬 RAG-Based Chatbot on Human Evolution (Wikipedia + LLaMA 3)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to build an intelligent chatbot capable of answering questions on **Human Evolution**. It uses Wikipedia as the knowledge base, **FAISS** for vector similarity search, **SentenceTransformers** for dense embedding generation, and **LLaMA 3** (served locally via **Ollama**) for context-aware generation. A comprehensive **evaluation pipeline** assesses answer quality using advanced NLP metrics.

---

## 📦 Project Highlights

* ✅ Local, private RAG pipeline with no external LLM API usage.
* 🔍 Embedding + FAISS for fast, semantic retrieval.
* 🤖 LLaMA 3 (8B) via Ollama for LLM generation.
* 📊 Multi-metric evaluation for rigorous benchmarking.

---

## 🧩 Components

### 1. 🧹 Web Scraping

* **Source**: Wikipedia pages related to **Human Evolution**.
* **Library Used**: `BeautifulSoup`, `requests`
* **Output**: `human_evolution_combined.txt` in the `data/` folder.

---

### 2. 📚 Embedding + FAISS Index Creation

* **Embedding Model**: `all-MiniLM-L6-v2` (or optionally `BAAI/bge-large-en-v1.5`)
* **Steps**:

  * Split text using `RecursiveCharacterTextSplitter`.
  * Convert chunks into dense vectors.
  * Store vectors in a **FAISS** index (`index.faiss` + `index.pkl`).
* **Output Directory**: `embeddings/`

---

### 3. 🧠 RAG Pipeline (Question Answering)

* **Input**: Questions from `generated_qa.csv`.
* **Retrieval**: Top-k context retrieval via FAISS.
* **LLM**: `llama3:8b` model served via Ollama.
* **Prompting**: Injects retrieved context into a fixed template prompt.
* **Output**: `qa_with_context.csv` containing:

  * `question`
  * `answer` (ground truth)
  * `retrieved_context`
  * `llm_answer` (model-generated)

---

### 4. 📏 Evaluation Pipeline

* **Script**: `evaluate_llm_answers.py`
* **Input**: `qa_with_context.csv`
* **Output**:

  * `qa_with_scores.csv` → scores per question
  * `qa_score_summary.csv` → aggregate results

#### ✅ Implemented Evaluation Metrics:

| Metric                  | Description                                                                  |
| ----------------------- | ---------------------------------------------------------------------------- |
| **BLEU**                | N-gram precision metric (1–4-grams) for fluency and lexical overlap.         |
| **ROUGE-1**             | Unigram recall-based score measuring overlap between LLM and reference.      |
| **ROUGE-L**             | Longest Common Subsequence similarity measure.                               |
| **BERTScore**           | Semantic similarity using contextual embeddings.                             |
| **BERT F1**             | F1 score computed via token similarity from BERT embeddings.                 |
| **F1-overlap**          | Classical token-level F1 based on exact match tokens.                        |
| **METEOR**              | Synonym- and paraphrase-aware metric using stemming and alignment.           |
| **Semantic Similarity** | Cosine similarity of sentence embeddings (SBERT).                            |
| **NLI Score / Label**   | Uses a Natural Language Inference model to detect entailment, contradiction. |

---

## 📂 Folder Structure

```
RAG_LLM/
├── data/
│   └── human_evolution_combined.txt          # Scraped Wikipedia text
│
├── embeddings/
│   ├── index.faiss                           # FAISS index
│   └── index.pkl                             # Document mapping
│
├── utils/
│   ├── scraper.py                            # Wikipedia scraper
│   └── ui_app.py                             # Streamlit interface (optional)
│
├── output_files/
│   ├── generated_qa.csv                      # Human-written QA pairs
│   ├── qa_with_context.csv                   # + Retrieved context + LLM answer
│   ├── qa_with_scores.csv                    # Evaluation scores per QA
│   └── qa_score_summary.csv                  # Aggregated evaluation results
│
├── src/
│   ├── generate_qa_dataset.py                # Script to create QA dataset
│   ├── generate_llm_answers.py               # RAG + LLM answering pipeline
│   ├── evaluate_llm_answers.py               # Metric evaluation script
│   └── rag_pipeline.py                       # Core FAISS + LLM orchestrator
│
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama with LLaMA 3

```bash
ollama run llama3
```

### 3. Generate LLM Answers (with Context)

```bash
python src/generate_llm_answers.py
```

### 4. Evaluate LLM Output

```bash
python src/evaluate_llm_answers.py
```

---

## 📁 Output Files

| File                   | Description                                    |
| ---------------------- | ---------------------------------------------- |
| `qa_with_context.csv`  | Combined data: question + context + LLM answer |
| `qa_with_scores.csv`   | Row-wise evaluation metrics for each QA        |
| `qa_score_summary.csv` | Mean/aggregate values for all metrics          |

---

## 🛠 Tech Stack

| Component          | Tool/Library                                                    |
| ------------------ | --------------------------------------------------------------- |
| Web Scraping       | `BeautifulSoup`, `requests`                                     |
| Embedding Models   | `SentenceTransformers`                                          |
| Vector DB          | `FAISS`                                                         |
| LLM Inference      | `LLaMA 3 (8B)` via `Ollama`                                     |
| Evaluation Metrics | `nltk`, `evaluate`, `scikit-learn`, `BERTScore`, `transformers` |
| Language           | Python 3.x                                                      |

---

