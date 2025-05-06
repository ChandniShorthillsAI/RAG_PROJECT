import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup

# Add the parent directory to sys.path to import the scraper module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.scraper import clean_text, scrape_wikipedia_page, save_data, TOPICS, SAVE_DIR

class TestScraper(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_text = "  This   is   a   test   text  "
        self.expected_clean_text = "This is a test text"
        self.test_title = "Human_evolution"
        self.test_url = f"https://en.wikipedia.org/wiki/{self.test_title}"

    def test_clean_text(self):
        """Test the clean_text function"""
        result = clean_text(self.test_text)
        self.assertEqual(result, self.expected_clean_text)

    @patch('requests.get')
    def test_scrape_wikipedia_page_success(self, mock_get):
        """Test successful Wikipedia page scraping"""
        # Mock HTML content
        mock_html = """
        <div class="mw-parser-output">
            <p>This is a test paragraph with more than 50 characters to ensure it gets included in the output.</p>
            <p>Short paragraph</p>
            <p>Another long paragraph with more than 50 characters to ensure it gets included in the output.</p>
        </div>
        """
        mock_response = MagicMock()
        mock_response.text = mock_html
        mock_get.return_value = mock_response

        result = scrape_wikipedia_page(self.test_title)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertNotIn("Short paragraph", result)

    @patch('requests.get')
    def test_scrape_wikipedia_page_failure(self, mock_get):
        """Test Wikipedia page scraping with network error"""
        mock_get.side_effect = Exception("Network error")
        result = scrape_wikipedia_page(self.test_title)
        self.assertEqual(result, "")

    def test_save_data(self):
        """Test the save_data function"""
        # Create a test topics list with just one topic
        test_topics = [self.test_title]
        
        # Mock the scrape_wikipedia_page function
        with patch('utils.scraper.scrape_wikipedia_page') as mock_scrape:
            mock_scrape.return_value = "Test content"
            
            # Call save_data
            save_data(test_topics)
            
            # Check if file was created
            file_path = os.path.join(SAVE_DIR, "modern_history_of_india.txt")
            self.assertTrue(os.path.exists(file_path))
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_topics_list(self):
        """Test if TOPICS list is properly defined"""
        self.assertIsInstance(TOPICS, list)
        self.assertGreater(len(TOPICS), 0)
        self.assertTrue(all(isinstance(topic, str) for topic in TOPICS))

if __name__ == '__main__':
    unittest.main() 