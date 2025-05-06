import unittest
import os
import sys
import pandas as pd
import json
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Create mock classes for FAISS and embeddings
class MockFAISS:
    def __init__(self):
        self.docstore = MagicMock()
        self.docstore._dict = {}

    def similarity_search(self, query, k=3):
        return [
            MagicMock(page_content="Mock content 1"),
            MagicMock(page_content="Mock content 2"),
            MagicMock(page_content="Mock content 3")
        ]

    @classmethod
    def load_local(cls, index_path, embeddings=None, **kwargs):
        return cls()

class MockEmbeddings:
    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [[0.1] * 384] * len(texts)

    def embed_query(self, text):
        return [0.1] * 384

# Mock the modules
sys.modules['langchain_community.vectorstores'] = MagicMock()
sys.modules['langchain_community.vectorstores'].FAISS = MockFAISS
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['langchain_huggingface'].HuggingFaceEmbeddings = MockEmbeddings

# Mock streamlit
sys.modules['streamlit'] = MagicMock()
sys.modules['streamlit'].cache_resource = lambda func: func

from src.evaluate_custom_llm_rag import (
    get_csv_path, load_csv_data, truncate, build_prompt,
    extract_json, evaluate_row
)

class TestEvaluateCustomLLMRag(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.test_dir, "output_files")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = {
            'question': ['What is AI?', 'What is ML?'],
            'retrieved_context': ['AI is artificial intelligence...', 'ML is machine learning...'],
            'answer': ['AI is artificial intelligence', 'ML is machine learning'],
            'llm_answer': ['AI stands for artificial intelligence', 'ML stands for machine learning']
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Save sample data to CSV
        self.input_csv = os.path.join(self.test_output_dir, 'qa_with_context.csv')
        self.df.to_csv(self.input_csv, index=False)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('pathlib.Path')
    def test_get_csv_path(self, mock_path):
        """Test get_csv_path function"""
        # Mock the path resolution
        mock_path.return_value.parent.parent = Path(self.test_dir)
        
        # Test path resolution
        csv_path = get_csv_path()
        self.assertIsInstance(csv_path, Path)
        self.assertTrue(str(csv_path).endswith('qa_with_context.csv'))

    def test_load_csv_data(self):
        """Test load_csv_data function"""
        # Test successful loading
        with patch('src.evaluate_custom_llm_rag.get_csv_path') as mock_get_path:
            mock_get_path.return_value = Path(self.input_csv)
            df = load_csv_data()
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            self.assertTrue(all(col in df.columns for col in self.sample_data.keys()))

        # Test file not found
        with patch('src.evaluate_custom_llm_rag.get_csv_path') as mock_get_path:
            mock_get_path.return_value = Path('nonexistent.csv')
            with self.assertRaises(FileNotFoundError):
                load_csv_data()

    def test_truncate(self):
        """Test truncate function"""
        # Test short text
        short_text = "Short text"
        self.assertEqual(truncate(short_text), short_text)
        
        # Test long text
        long_text = "This is a very long text that should be truncated" * 10
        truncated = truncate(long_text)
        self.assertLessEqual(len(truncated), 103)  # 100 + 3 for "..."
        self.assertTrue(truncated.endswith("..."))

    def test_build_prompt(self):
        """Test build_prompt function"""
        question = "What is AI?"
        context = "AI is artificial intelligence"
        answer = "AI is artificial intelligence"
        llm_answer = "AI stands for artificial intelligence"
        
        prompt = build_prompt(question, context, answer, llm_answer)
        
        # Verify prompt structure
        self.assertIn(question, prompt)
        self.assertIn(context, prompt)
        self.assertIn(answer, prompt)
        self.assertIn(llm_answer, prompt)
        self.assertIn("Correctness", prompt)
        self.assertIn("Relevance", prompt)
        self.assertIn("Similarity", prompt)
        self.assertIn("Context Utilization", prompt)
        self.assertIn("Faithfulness", prompt)

    def test_extract_json(self):
        """Test extract_json function"""
        # Test valid JSON
        valid_json = '{"correctness": 0.8, "relevance": 0.9, "similarity": 0.85, "context_utilization": 0.75, "faithfulness": 0.8, "explanation": "Good answer"}'
        result = extract_json(valid_json)
        self.assertEqual(result["correctness"], 0.8)
        self.assertEqual(result["relevance"], 0.9)
        self.assertEqual(result["explanation"], "Good answer")
        
        # Test invalid JSON
        invalid_json = "Not a JSON"
        result = extract_json(invalid_json)
        self.assertEqual(result["correctness"], 0.0)
        self.assertEqual(result["explanation"], "No JSON found")
        
        # Test malformed JSON
        malformed_json = "{correctness: 0.8}"  # Missing quotes
        result = extract_json(malformed_json)
        self.assertEqual(result["correctness"], 0.0)
        self.assertEqual(result["explanation"], "Invalid JSON")

    @patch('src.evaluate_custom_llm_rag.get_llm_score')
    def test_evaluate_row(self, mock_get_llm_score):
        """Test evaluate_row function"""
        # Mock LLM response
        mock_response = {
            'correctness': 0.8,
            'relevance': 0.9,
            'similarity': 0.85,
            'context_utilization': 0.75,
            'faithfulness': 0.8,
            'explanation': 'Good answer'
        }
        mock_get_llm_score.return_value = mock_response
        
        # Test evaluation
        row = self.df.iloc[0]
        i, result = evaluate_row(0, row)
        
        # Verify results
        self.assertEqual(i, 0)
        self.assertEqual(result['llm_correctness'], 0.8)
        self.assertEqual(result['llm_relevance'], 0.9)
        self.assertEqual(result['llm_explanation'], 'Good answer')
        
        # Test already evaluated row
        row['llm_correctness'] = 0.5
        i, result = evaluate_row(0, row)
        self.assertEqual(i, 0)
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main() 