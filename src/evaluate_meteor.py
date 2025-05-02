import pandas as pd
from nltk.translate.meteor_score import meteor_score
import nltk
import os

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('punkt')

# File paths
input_path = "output_files/qa_with_context.csv"
output_path = "output_files/qa_with_meteor.csv"

# Check if file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"File not found: {input_path}")

# Load the CSV
df = pd.read_csv(input_path)

# Compute METEOR scores
meteor_scores = []
for i, row in df.iterrows():
    reference = [row['ground_truth']]
    prediction = row['llm_answer']
    score = meteor_score(reference, prediction)
    meteor_scores.append(score)

# Add scores and save the new CSV
df['meteor_score'] = meteor_scores
df.to_csv(output_path, index=False)

print(f"✅ Evaluation complete. METEOR scores saved to: {output_path}")
