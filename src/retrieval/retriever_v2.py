"""
Enhanced retriever with query expansion
"""

from typing import List, Tuple
from pathlib import Path
import numpy as np

from src.retrieval.embedder import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.ingestion.chunker import SemanticChunk
from src.retrieval.query_expander import QueryExpander
from src.config import VECTOR_STORE_DIR, TOP_K_RESULTS


class EnhancedRetriever:
    """Retriever with query expansion and fusion"""
    
    def __init__(self, vector_store: FAISSVectorStore, embedder: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_expander = QueryExpander()
    
    def retrieve_with_expansion(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[SemanticChunk, float]]:
        """
        Retrieve using query expansion and fusion
        """
        # Expand query
        queries = self.query_expander.expand_query(query)
        
        # Retrieve for each query variant
        all_results = {}  # chunk_id -> (chunk, max_score)
        
        for q in queries:
            # Generate embedding
            query_embedding = self.embedder.model.encode(
                [q],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Search
            results = self.vector_store.search(query_embedding, top_k=top_k * 2)
            
            # Merge results (keep highest score for each chunk)
            for chunk, score in results:
                chunk_id = chunk.chunk_id
                if chunk_id not in all_results or score > all_results[chunk_id][1]:
                    all_results[chunk_id] = (chunk, score)
        
        # Convert back to list and sort
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda x: x[1], reverse=True)
        
        return merged_results[:top_k]
    
    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[SemanticChunk, float]]:
        """Standard retrieval"""
        query_embedding = self.embedder.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return self.vector_store.search(query_embedding, top_k=top_k)
    
    @classmethod
    def load_from_disk(cls, load_dir: Path = VECTOR_STORE_DIR):
        """Load from disk"""
        print("Loading enhanced retrieval system...")
        embedder = EmbeddingGenerator()
        vector_store = FAISSVectorStore.load(load_dir)
        print("Enhanced retrieval system loaded")
        return cls(vector_store, embedder)

