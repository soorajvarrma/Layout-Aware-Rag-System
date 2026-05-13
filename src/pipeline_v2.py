"""
Updated QA pipeline with hybrid retrieval
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.client import LLMClient
from src.llm.generator import AnswerGenerator
from src.ingestion.chunker import SemanticChunk
from src.config import VECTOR_STORE_DIR


class RetrieverWrapper:
    """Wrapper to make HybridRetriever compatible with AnswerGenerator"""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.backend = hybrid_retriever
        self.vector_store = hybrid_retriever.vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[SemanticChunk, float]]:
        """Standard retrieve"""
        return self.backend.retrieve(query, top_k)
    
    def retrieve_with_reranking(self, query: str, top_k: int = 5, 
                               initial_k: int = 20) -> List[Tuple[SemanticChunk, float]]:
        """Retrieve with reranking - just use top_k since hybrid does its own thing"""
        return self.backend.retrieve(query, top_k)


class ImprovedQAPipeline:
    """QA Pipeline with hybrid retrieval"""
    
    def __init__(self, vector_store_dir: Path = VECTOR_STORE_DIR):
        print("Initializing improved QA Pipeline...")
        
        # Load hybrid retrieval system
        hybrid_retriever = HybridRetriever.load_from_disk(vector_store_dir)
        
        # Wrap for compatibility
        self.retriever = RetrieverWrapper(hybrid_retriever)
        
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Create answer generator
        self.generator = AnswerGenerator(self.retriever, self.llm_client)
        
        print("Improved QA Pipeline ready")
    
    def ask(self, question: str, top_k: int = 7) -> Dict[str, Any]:
        """Ask a question - using more chunks by default"""
        return self.generator.generate_answer(question, top_k=top_k)
    
    def ask_multiple(self, questions: list, top_k: int = 7) -> list:
        """Ask multiple questions"""
        return self.generator.batch_generate(questions, top_k=top_k)
