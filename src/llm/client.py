"""
LLM client for answer generation
"""

from groq import Groq
from typing import List, Dict, Optional
from src.config import GROQ_API_KEY, LLM_MODEL


class LLMClient:
    """Groq API client for answer generation"""
    
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = LLM_MODEL):
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"Initialized LLM client with model: {model}")
    
    def generate(self, messages: List[Dict[str, str]], 
                temperature: float = 0.1,
                max_tokens: int = 1024) -> str:
        """
        Generate response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer given query and context
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        messages = [
            {
                "role": "system",
                "content": """You are a precise AI assistant specialized in answering questions about academic research papers.

Your task:
1. Answer the question using ONLY information from the provided context
2. Be accurate and specific - cite exact details from the context
3. If the context contains tables, reference specific numbers and comparisons
4. If the answer is not in the context, say "The provided context does not contain this information"
5. Keep answers concise but complete
6. Do not add information from your general knowledge

Format:
- Direct answers for factual questions
- Quote relevant parts when appropriate
- Reference table/figure numbers when applicable"""
            },
            {
                "role": "user",
                "content": f"""Context from the BERT paper:

{context}

Question: {query}

Answer:"""
            }
        ]
        
        return self.generate(messages, temperature=0.1)
