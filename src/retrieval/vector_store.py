"""
FAISS-based vector store for semantic search
"""

import faiss
import numpy as np
from typing import List, Tuple
from pathlib import Path
import pickle

from src.ingestion.chunker import SemanticChunk


class FAISSVectorStore:
    """Vector store using FAISS for fast similarity search"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS index
        
        Args:
            embedding_dim: Dimension of embeddings (384 for MiniLM)
        """
        self.embedding_dim = embedding_dim
        
        # Use L2 distance (works well with normalized embeddings)
        # For cosine similarity with normalized vectors, L2 is equivalent
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Store chunks for retrieval
        self.chunks: List[SemanticChunk] = []
        
        print(f"Initialized FAISS index (dim={embedding_dim})")
    
    def add_documents(self, chunks: List[SemanticChunk], embeddings: np.ndarray):
        """
        Add documents to the vector store
        
        Args:
            chunks: List of SemanticChunk objects
            embeddings: Numpy array of embeddings (n_chunks, embedding_dim)
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunks)} chunks")
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        self.chunks = chunks
        
        print(f" Added {len(chunks)} documents to vector store")
        print(f"   Total vectors in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[SemanticChunk, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding (1, embedding_dim)
            top_k: Number of results to return
            
        Returns:
            List of (chunk, distance) tuples
        """
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return chunks with scores
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                # Convert L2 distance to similarity score (inverse)
                similarity = 1 / (1 + dist)
                results.append((self.chunks[idx], similarity))
        
        return results
    
    def save(self, save_dir: Path):
        """Save vector store to disk"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(faiss_path))
        
        # Save chunks
        chunks_path = save_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved vector store to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Path) -> 'FAISSVectorStore':
        """Load vector store from disk"""
        # Load FAISS index
        faiss_path = load_dir / "faiss_index.bin"
        index = faiss.read_index(str(faiss_path))
        
        # Load chunks
        chunks_path = load_dir / "chunks.pkl"
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Create instance
        vector_store = cls(embedding_dim=index.d)
        vector_store.index = index
        vector_store.chunks = chunks
        
        print(f"Loaded vector store from {load_dir}")
        print(f"   Total vectors: {index.ntotal}")
        
        return vector_store
