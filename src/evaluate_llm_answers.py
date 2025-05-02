

# import pandas as pd
# from tqdm import tqdm
# from bert_score import score as bert_score
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
# from fuzzywuzzy import fuzz
# import numpy as np

# # Load the CSV file
# df = pd.read_csv("qa_with_context.csv")

# # Filter out rows with empty values
# df = df.dropna(subset=["answer", "llm_answer"])

# # Initialize sentence transformer model
# sent_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Compute BERTScore
# print("🔍 Calculating BERTScore...")
# P, R, F1 = bert_score(df["llm_answer"].tolist(), df["answer"].tolist(), lang="en", model_type="roberta-large")

# # Compute cosine similarity
# print("🔍 Calculating Cosine Similarity...")
# ref_embeddings = sent_model.encode(df["answer"].tolist(), convert_to_tensor=True)
# pred_embeddings = sent_model.encode(df["llm_answer"].tolist(), convert_to_tensor=True)
# cosine_scores = cosine_similarity(ref_embeddings, pred_embeddings)
# cosine_diag = cosine_scores.diagonal()

# # Compute fuzzy similarity
# print("🔍 Calculating Fuzzy Match Score...")
# fuzzy_scores = [fuzz.token_set_ratio(a, b) for a, b in zip(df["answer"], df["llm_answer"])]

# # Print results
# print("\n📊 Evaluation Results:")
# print(f"BERTScore - Precision: {P.mean().item():.4f}, Recall: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")
# print(f"Cosine Similarity (Sentence Embedding): {np.mean(cosine_diag):.4f}")
# print(f"Fuzzy Token Set Ratio: {np.mean(fuzzy_scores):.2f}")



import pandas as pd
from tqdm import tqdm
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import numpy as np

# Load the CSV file
df = pd.read_csv("qa_with_context.csv")

# Filter out rows with empty values
df = df.dropna(subset=["answer", "llm_answer"]).reset_index(drop=True)

# Initialize sentence transformer model
sent_model = SentenceTransformer("all-MiniLM-L6-v2")


# Compute BERTScore
print("🔍 Calculating BERTScore...")
P, R, F1 = bert_score(df["llm_answer"].tolist(), df["answer"].tolist(), lang="en", model_type="roberta-large")

# Compute cosine similarity
print("🔍 Calculating Cosine Similarity...")
ref_embeddings = sent_model.encode(df["answer"].tolist(), convert_to_tensor=True)
pred_embeddings = sent_model.encode(df["llm_answer"].tolist(), convert_to_tensor=True)
cosine_scores = cosine_similarity(ref_embeddings, pred_embeddings)
cosine_diag = cosine_scores.diagonal()

# Compute fuzzy similarity
print("🔍 Calculating Fuzzy Match Score...")
fuzzy_scores = [fuzz.token_set_ratio(a, b) for a, b in zip(df["answer"], df["llm_answer"])]

# Add scores to DataFrame
df["bert_precision"] = [p.item() for p in P]
df["bert_recall"] = [r.item() for r in R]
df["bert_f1"] = [f.item() for f in F1]
df["cosine_similarity"] = cosine_diag
df["fuzzy_score"] = fuzzy_scores

# Save detailed scores
df.to_csv("qa_with_scores.csv", index=False)

# Compute summary
summary = {
    "Average BERT Precision": [P.mean().item()],
    "Average BERT Recall": [R.mean().item()],
    "Average BERT F1": [F1.mean().item()],
    "Average Cosine Similarity": [np.mean(cosine_diag)],
    "Average Fuzzy Score": [np.mean(fuzzy_scores)],
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv("qa_score_summary.csv", index=False)

# Print summary
print("\n📊 Evaluation Summary:")
print(summary_df.to_string(index=False))
print("✅ Scores saved to qa_with_scores.csv and qa_score_summary.csv")
