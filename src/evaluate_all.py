import pandas as pd
import os
from tqdm import tqdm

# Your evaluate_response() function here
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import f1_score
import bert_score
from transformers import pipeline
import nltk

nltk.download('punkt')
nltk.download('wordnet')

# Models
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_response(true_answer, chatbot_response, model=semantic_model):
    scores = {}

    smoothie = SmoothingFunction().method4
    reference = [true_answer.split()]
    candidate = chatbot_response.split()
    scores['bleu'] = sentence_bleu(reference, candidate, smoothing_function=smoothie)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(true_answer, chatbot_response)
    scores['rouge1'] = rouge_scores['rouge1'].fmeasure
    scores['rougeL'] = rouge_scores['rougeL'].fmeasure

    scores['meteor'] = single_meteor_score(true_answer, chatbot_response)

    emb1 = model.encode(true_answer, convert_to_tensor=True)
    emb2 = model.encode(chatbot_response, convert_to_tensor=True)
    scores['semantic_similarity'] = util.pytorch_cos_sim(emb1, emb2).item()

    P, R, F1 = bert_score.score([chatbot_response], [true_answer], lang="en", verbose=False)
    scores['bert_f1'] = F1[0].item()

    true_tokens = true_answer.split()
    pred_tokens = chatbot_response.split()
    common_tokens = set(true_tokens) & set(pred_tokens)

    if len(true_tokens) > 0 and len(pred_tokens) > 0:
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        scores['f1_overlap'] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    else:
        scores['f1_overlap'] = 0.0

    try:
        nli_output = nli_model(f"{true_answer} </s> {chatbot_response}", truncation=True)
        scores['nli_label'] = nli_output[0]['label']
        scores['nli_score'] = nli_output[0]['score']
    except Exception as e:
        scores['nli_label'] = 'Error'
        scores['nli_score'] = 0.0

    return scores

# Evaluation Runner
df = pd.read_csv("../output_files/qa_with_context.csv")
all_scores = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    true_answer = str(row['answer'])
    chatbot_response = str(row['llm_answer'])
    scores = evaluate_response(true_answer, chatbot_response)
    all_scores.append(scores)

# Save individual scores
scores_df = pd.DataFrame(all_scores)
final_df = pd.concat([df, scores_df], axis=1)
final_df.to_csv("../output_files/llm_metrics.csv", index=False)

# Save average metrics
average_scores = scores_df.mean(numeric_only=True)
average_scores.to_frame(name="average_score").to_csv("../output_files/llm_metrics_summary.csv")

print("Evaluation completed. Metrics saved to:")
print("- output_files/llm_metrics.csv")
print("- output_files/llm_metrics_summary.csv")
