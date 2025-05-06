import unittest
import os
import sys
import shutil
from unittest.mock import patch, MagicMock
import tempfile

# Add the parent directory to sys.path to import the RAG pipeline module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag_pipeline import RagPipeline

class TestRagPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.test_txt_path = os.path.join(self.test_dir, "test.txt")
        self.test_index_path = os.path.join(self.test_dir, "test_index")
        
        # Create a sample text file
        with open(self.test_txt_path, "w", encoding="utf-8") as f:
            f.write("This is a test document[1]. It contains some text[citation needed].\n\n"
                   "This is another paragraph with more text. It should be split into chunks.\n\n"
                   "This is a third paragraph that will be used for testing the chunking functionality.")
        
        self.pipeline = RagPipeline(
            txt_file_path=self.test_txt_path,
            index_path=self.test_index_path,
            chunk_size=100,
            chunk_overlap=20
        )

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.test_dir)

    def test_clean_text(self):
        """Test the clean_text function"""
        test_text = "This is a test[1] with [citation needed] and extra   spaces"
        expected = "This is a test with and extra spaces"
        result = self.pipeline.clean_text(test_text)
        self.assertEqual(result, expected)

    def test_load_and_chunk_text(self):
        """Test the load_and_chunk_text function"""
        documents = self.pipeline.load_and_chunk_text()
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(hasattr(doc, 'page_content') for doc in documents))
        self.assertTrue(all(hasattr(doc, 'metadata') for doc in documents))

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('langchain_community.vectorstores.FAISS')
    def test_embed_and_store(self, mock_faiss, mock_embeddings):
        """Test the embed_and_store function"""
        # Mock the embedding model
        mock_embeddings.return_value.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock FAISS
        mock_faiss.from_documents.return_value = MagicMock()
        
        # Create test documents
        documents = self.pipeline.load_and_chunk_text()
        
        # Test embedding and storing
        self.pipeline.embed_and_store(documents)
        
        # Verify FAISS was called
        mock_faiss.from_documents.assert_called_once()

    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    @patch('langchain_community.vectorstores.FAISS')
    def test_retrieve_context(self, mock_faiss, mock_embeddings):
        """Test the retrieve_context function"""
        # Mock the embedding model
        mock_embeddings.return_value = MagicMock()
        
        # Mock FAISS and its similarity search
        mock_faiss_instance = MagicMock()
        mock_faiss.load_local.return_value = mock_faiss_instance
        mock_faiss_instance.similarity_search.return_value = [
            MagicMock(page_content="Test content 1"),
            MagicMock(page_content="Test content 2")
        ]
        
        # Test retrieval
        result = self.pipeline.retrieve_context("test query", k=2)
        
        # Verify the result
        self.assertIsInstance(result, str)
        self.assertIn("Test content 1", result)
        self.assertIn("Test content 2", result)

    def test_pipeline_initialization(self):
        """Test the pipeline initialization"""
        self.assertEqual(self.pipeline.txt_file_path, self.test_txt_path)
        self.assertEqual(self.pipeline.index_path, self.test_index_path)
        self.assertEqual(self.pipeline.chunk_size, 100)
        self.assertEqual(self.pipeline.chunk_overlap, 20)
        self.assertIsNotNone(self.pipeline.splitter)
        self.assertIsNotNone(self.pipeline.embedding_model)

if __name__ == '__main__':
    unittest.main() 