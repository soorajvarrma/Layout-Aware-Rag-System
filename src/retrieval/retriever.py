"""
High-level retrieval pipeline with improved strategies
"""

from typing import List, Tuple
from pathlib import Path

from src.retrieval.embedder import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.ingestion.chunker import SemanticChunk
from src.config import VECTOR_STORE_DIR, TOP_K_RESULTS


class SemanticRetriever:
    """Complete retrieval pipeline"""
    
    def __init__(self, vector_store: FAISSVectorStore, embedder: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Tuple[SemanticChunk, float]]:
        """
        Retrieve relevant chunks for a query
        """
        # Generate query embedding
        query_embedding = self.embedder.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    def retrieve_with_reranking(self, query: str, top_k: int = TOP_K_RESULTS, 
                                initial_k: int = 30) -> List[Tuple[SemanticChunk, float]]:
        """
        Retrieve with two-stage approach: initial retrieval + reranking
        Retrieves more candidates initially for better coverage
        """
        # Stage 1: Retrieve more candidates
        candidates = self.retrieve(query, top_k=initial_k)
        
        # Stage 2: Filter out very short chunks (likely headers without content)
        filtered = []
        for chunk, score in candidates:
            # Skip very short text chunks (likely just headers)
            if chunk.chunk_type == 'text' and len(chunk.text.strip()) < 30:
                # But reduce score rather than skip entirely
                filtered.append((chunk, score * 0.5))
            else:
                filtered.append((chunk, score))
        
        # Stage 3: Rerank by chunk type priority
        reranked = self._rerank_by_type(filtered, query)
        
        return reranked[:top_k]
    
    def _rerank_by_type(self, candidates: List[Tuple[SemanticChunk, float]], 
                        query: str) -> List[Tuple[SemanticChunk, float]]:
        """Enhanced reranking based on chunk type and query"""
        
        boosted = []
        query_lower = query.lower()
        
        for chunk, score in candidates:
            boost = 1.0
            
            # Boost substantive text chunks
            if chunk.chunk_type == 'text' and len(chunk.text) > 100:
                boost *= 1.1
            
            # Boost tables if query asks for numbers/results/performance
            if chunk.chunk_type == 'table':
                if any(word in query_lower for word in 
                       ['score', 'result', 'performance', 'accuracy', 'compare', 
                        'benchmark', 'number', 'percentage']):
                    boost *= 1.3
            
            # Boost figures if query asks about architecture/diagram
            if chunk.chunk_type == 'figure':
                if any(word in query_lower for word in 
                       ['architecture', 'diagram', 'figure', 'model', 'structure',
                        'illustration', 'visualization']):
                    boost *= 1.2
            
            # Boost if query keywords appear in chunk
            query_words = set(query_lower.split())
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 2:
                boost *= (1 + overlap * 0.05)
            
            boosted.append((chunk, score * boost))
        
        # Sort by boosted score
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted
    
    @classmethod
    def build_from_chunks(cls, chunks: List[SemanticChunk], 
                         save_dir: Path = VECTOR_STORE_DIR) -> 'SemanticRetriever':
        """Build retriever from scratch"""
        print("Building retrieval system...")
        
        embedder = EmbeddingGenerator()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.generate_embeddings(texts)
        
        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        vector_store.add_documents(chunks, embeddings)
        vector_store.save(save_dir)
        
        print("Retrieval system built successfully")
        
        return cls(vector_store, embedder)
    
    @classmethod
    def load_from_disk(cls, load_dir: Path = VECTOR_STORE_DIR) -> 'SemanticRetriever':
        """Load pre-built retriever from disk"""
        print("Loading retrieval system...")
        
        embedder = EmbeddingGenerator()
        vector_store = FAISSVectorStore.load(load_dir)
        
        print("Retrieval system loaded")
        
        return cls(vector_store, embedder)
