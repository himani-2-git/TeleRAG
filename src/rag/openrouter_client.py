"""OpenRouter API client for LLM interactions."""

import requests
from typing import List, Dict, Any, Optional
import logging
import time

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "qwen/qwen3.8-27b:free",
                 max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier to use
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries in seconds
        """
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def send_request(self, messages: List[Dict[str, str]], 
                    temperature: float = 0.7) -> Optional[str]:
        """Send chat completion request to OpenRouter.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            
        Returns:
            Generated response text or None if failed
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._extract_response(data)
                
                elif response.status_code == 429:
                    # Rate limit hit
                    logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))
                        continue
                    return None
                
                else:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                return None
        
        return None
    
    def _extract_response(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract response text from API response.
        
        Args:
            data: API response data
            
        Returns:
            Response text or None
        """
        try:
            return data['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract response: {e}")
            return None
    
    def format_prompt(self, question: str, context: str) -> List[Dict[str, str]]:
        """Format question and context into prompt messages.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            List of message dictionaries
        """
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions based on provided document context. "
                "Use only the information from the context to answer questions. "
                "If the context doesn't contain relevant information, say so clearly. "
                "Be concise and accurate."
            )
        }
        
        user_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        }
        
        return [system_message, user_message]
    
    def generate_answer(self, question: str, context: str) -> Optional[str]:
        """Generate answer for a question using context.
        
        Args:
            question: User's question
            context: Retrieved document context
            
        Returns:
            Generated answer or None
        """
        if not context:
            return "I don't have any relevant information to answer this question. Please upload a PDF document first."
        
        messages = self.format_prompt(question, context)
        return self.send_request(messages)
    
    def validate_response(self, response: Optional[str]) -> str:
        """Validate and clean response.
        
        Args:
            response: Raw response from API
            
        Returns:
            Cleaned response
        """
        if not response:
            return "I'm sorry, I couldn't generate a response at this time. Please try again."
        
        # Basic cleaning
        response = response.strip()
        
        return response
