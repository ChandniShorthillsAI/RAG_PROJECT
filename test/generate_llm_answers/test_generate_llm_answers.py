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

# Mock constants
with patch('src.generate_llm_answers.INPUT_CSV', 'generated_qa.csv'), \
     patch('src.generate_llm_answers.OUTPUT_CSV', 'qa_with_context.csv'):
    from src.generate_llm_answers import (
        stream_ollama, get_done_questions, append_to_csv, generate_llm_answers
    )

class TestGenerateLLMAnswers(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_output_dir = os.path.join(self.test_dir, "output_files")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Create sample data
        self.sample_data = {
            'question': ['What is AI?', 'What is ML?'],
            'answer': ['AI is artificial intelligence', 'ML is machine learning']
        }
        self.df = pd.DataFrame(self.sample_data)
        
        # Save sample data to CSV
        self.input_csv = os.path.join(self.test_output_dir, 'generated_qa.csv')
        self.output_csv = os.path.join(self.test_output_dir, 'qa_with_context.csv')
        self.df.to_csv(self.input_csv, index=False)

        # Patch the CSV paths
        self.path_patch = patch('src.generate_llm_answers.INPUT_CSV', self.input_csv)
        self.path_patch.start()
        self.output_patch = patch('src.generate_llm_answers.OUTPUT_CSV', self.output_csv)
        self.output_patch.start()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.path_patch.stop()
        self.output_patch.stop()
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('requests.post')
    def test_stream_ollama(self, mock_post):
        """Test stream_ollama function"""
        # Mock response
        mock_response = MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps({"response": "Hello"}).encode(),
            json.dumps({"response": " World"}).encode()
        ]
        mock_post.return_value = mock_response
        
        # Test streaming
        result = stream_ollama("Test prompt")
        self.assertEqual(result, "Hello World")
        
        # Test error handling
        mock_post.side_effect = Exception("API Error")
        result = stream_ollama("Test prompt")
        self.assertEqual(result, "")

    def test_get_done_questions(self):
        """Test get_done_questions function"""
        # Test empty file
        result = get_done_questions()
        self.assertEqual(result, set())
        
        # Test with existing data
        test_data = pd.DataFrame({
            'question': ['Q1', 'Q2'],
            'answer': ['A1', 'A2'],
            'retrieved_context': ['C1', 'C2'],
            'llm_answer': ['LA1', 'LA2']
        })
        test_data.to_csv(self.output_csv, index=False)
        
        result = get_done_questions()
        self.assertEqual(result, {'Q1', 'Q2'})

    def test_append_to_csv(self):
        """Test append_to_csv function"""
        # Test appending to empty file
        data = [{
            'question': 'Q1',
            'answer': 'A1',
            'retrieved_context': 'C1',
            'llm_answer': 'LA1'
        }]
        append_to_csv(data)
        
        # Verify file contents
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['question'], 'Q1')
        
        # Test appending to existing file
        data = [{
            'question': 'Q2',
            'answer': 'A2',
            'retrieved_context': 'C2',
            'llm_answer': 'LA2'
        }]
        append_to_csv(data)
        
        # Verify updated contents
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['question'], 'Q2')

    @patch('src.generate_llm_answers.stream_ollama')
    @patch('src.generate_llm_answers.vectorstore')
    def test_generate_llm_answers(self, mock_vectorstore, mock_stream):
        """Test generate_llm_answers function"""
        # Mock vectorstore
        mock_vectorstore.similarity_search.return_value = [
            MagicMock(page_content="Mock context 1"),
            MagicMock(page_content="Mock context 2")
        ]
        
        # Mock LLM response
        mock_stream.return_value = "This is a test answer"
        
        # Run generation
        generate_llm_answers()
        
        # Verify output file
        self.assertTrue(os.path.exists(self.output_csv))
        df = pd.read_csv(self.output_csv)
        self.assertEqual(len(df), 2)  # Should process both questions
        self.assertTrue('llm_answer' in df.columns)
        self.assertTrue('retrieved_context' in df.columns)

if __name__ == '__main__':
    unittest.main() 