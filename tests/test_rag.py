"""Tests for RAG engine functionality."""

import unittest
from unittest.mock import Mock, patch
from src.rag.embeddings import EmbeddingGenerator
from src.rag.retriever import DocumentRetriever
from src.rag.openrouter_client import OpenRouterClient

class TestEmbeddingGenerator(unittest.TestCase):
    """Test embedding generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a small model for testing
        self.generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    
    def test_generate_single_embedding(self):
        """Test generating single embedding."""
        text = "This is a test sentence."
        embedding = self.generator.generate_embeddings(text)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)
    
    def test_generate_multiple_embeddings(self):
        """Test generating multiple embeddings."""
        texts = ["First sentence.", "Second sentence."]
        embeddings = self.generator.generate_embeddings(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], list)
        self.assertIsInstance(embeddings[1], list)

class TestDocumentRetriever(unittest.TestCase):
    """Test document retriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retriever = DocumentRetriever(top_k=3, similarity_threshold=0.5)
    
    def test_search_similar(self):
        """Test similarity search."""
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        chunks = [
            {
                'content': 'Test chunk 1',
                'embedding': [0.1, 0.2, 0.3, 0.4],
                'filename': 'test.pdf'
            },
            {
                'content': 'Test chunk 2',
                'embedding': [0.9, 0.8, 0.7, 0.6],
                'filename': 'test.pdf'
            }
        ]
        
        results = self.retriever.search_similar(query_embedding, chunks)
        
        self.assertGreater(len(results), 0)
        self.assertIn('similarity_score', results[0])
    
    def test_format_context(self):
        """Test context formatting."""
        chunks = [
            {
                'content': 'Test content',
                'filename': 'test.pdf',
                'similarity_score': 0.95
            }
        ]
        
        context = self.retriever.format_context(chunks)
        
        self.assertIn('Test content', context)
        self.assertIn('test.pdf', context)

class TestOpenRouterClient(unittest.TestCase):
    """Test OpenRouter client."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = OpenRouterClient(api_key="test_key", model="test_model")
    
    def test_format_prompt(self):
        """Test prompt formatting."""
        question = "What is this about?"
        context = "This is test context."
        
        messages = self.client.format_prompt(question, context)
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertIn(question, messages[1]['content'])
        self.assertIn(context, messages[1]['content'])
    
    def test_validate_response(self):
        """Test response validation."""
        response = "  Test response  "
        validated = self.client.validate_response(response)
        
        self.assertEqual(validated, "Test response")
    
    def test_validate_none_response(self):
        """Test validation of None response."""
        validated = self.client.validate_response(None)
        
        self.assertIn("couldn't generate", validated)
    
    @patch('requests.post')
    def test_send_request_success(self, mock_post):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'Test answer'}}]
        }
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'Test'}]
        response = self.client.send_request(messages)
        
        self.assertEqual(response, 'Test answer')
    
    @patch('requests.post')
    def test_send_request_failure(self, mock_post):
        """Test failed API request."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response
        
        messages = [{'role': 'user', 'content': 'Test'}]
        response = self.client.send_request(messages)
        
        self.assertIsNone(response)

if __name__ == '__main__':
    unittest.main()
