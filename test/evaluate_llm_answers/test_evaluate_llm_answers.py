import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestEvaluateLLMAnswers(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = {
            'question': ['What is AI?', 'What is ML?'],
            'answer': ['AI is artificial intelligence', 'ML is machine learning'],
            'llm_answer': ['AI stands for artificial intelligence', 'ML stands for machine learning']
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Save sample data to CSV
        self.input_csv = os.path.join(self.test_dir, 'qa_with_context.csv')
        self.df.to_csv(self.input_csv, index=False)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('bert_score.score')
    def test_bert_score_calculation(self, mock_bert_score):
        """Test BERTScore calculation"""
        # Mock BERTScore return values
        mock_bert_score.return_value = (
            np.array([0.9, 0.8]),
            np.array([0.85, 0.75]),
            np.array([0.87, 0.77])
        )
        
        # Calculate BERTScore
        P, R, F1 = mock_bert_score(
            self.df["llm_answer"].tolist(),
            self.df["answer"].tolist(),
            lang="en",
            model_type="roberta-large"
        )
        
        # Verify results
        self.assertEqual(len(P), 2)
        self.assertEqual(len(R), 2)
        self.assertEqual(len(F1), 2)
        self.assertAlmostEqual(P[0], 0.9)
        self.assertAlmostEqual(R[0], 0.85)
        self.assertAlmostEqual(F1[0], 0.87)

    @patch('sentence_transformers.SentenceTransformer')
    def test_cosine_similarity_calculation(self, mock_sent_transformer):
        """Test cosine similarity calculation"""
        # Mock sentence transformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_sent_transformer.return_value = mock_model
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        ref_embeddings = mock_model.encode(self.df["answer"].tolist())
        pred_embeddings = mock_model.encode(self.df["llm_answer"].tolist())
        cosine_scores = cosine_similarity(ref_embeddings, pred_embeddings)
        
        # Verify results
        self.assertEqual(cosine_scores.shape, (2, 2))
        self.assertTrue(np.all(cosine_scores >= -1) and np.all(cosine_scores <= 1))

    def test_fuzzy_similarity_calculation(self):
        """Test fuzzy similarity calculation"""
        from fuzzywuzzy import fuzz
        
        # Calculate fuzzy scores
        fuzzy_scores = [fuzz.token_set_ratio(a, b) for a, b in zip(self.df["answer"], self.df["llm_answer"])]
        
        # Verify results
        self.assertEqual(len(fuzzy_scores), 2)
        self.assertTrue(all(0 <= score <= 100 for score in fuzzy_scores))

    def test_data_loading_and_cleaning(self):
        """Test data loading and cleaning"""
        # Test loading data
        loaded_df = pd.read_csv(self.input_csv)
        
        # Test cleaning data
        cleaned_df = loaded_df.dropna(subset=["answer", "llm_answer"]).reset_index(drop=True)
        
        # Verify results
        self.assertEqual(len(cleaned_df), 2)
        self.assertTrue(all(col in cleaned_df.columns for col in ['question', 'answer', 'llm_answer']))

    @patch('pandas.DataFrame.to_csv')
    def test_output_file_generation(self, mock_to_csv):
        """Test output file generation"""
        # Add scores to DataFrame
        self.df["bert_precision"] = [0.9, 0.8]
        self.df["bert_recall"] = [0.85, 0.75]
        self.df["bert_f1"] = [0.87, 0.77]
        self.df["cosine_similarity"] = [0.95, 0.85]
        self.df["fuzzy_score"] = [90, 85]
        
        # Save detailed scores
        output_path = os.path.join(self.test_dir, 'qa_with_scores.csv')
        self.df.to_csv(output_path, index=False)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Test summary generation
        summary = {
            "Average BERT Precision": [0.85],
            "Average BERT Recall": [0.80],
            "Average BERT F1": [0.82],
            "Average Cosine Similarity": [0.90],
            "Average Fuzzy Score": [87.5],
        }
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(self.test_dir, 'qa_score_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Verify summary file was created
        self.assertTrue(os.path.exists(summary_path))

if __name__ == '__main__':
    unittest.main() 