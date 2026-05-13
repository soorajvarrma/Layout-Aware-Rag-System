"""
Generate embeddings for semantic search
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

from src.config import EMBEDDING_MODEL, PROCESSED_DATA_DIR


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f" Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        print(f" Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"Generated embeddings: {embeddings.shape}")
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, save_path: Path):
        """Save embeddings to disk"""
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f" Saved embeddings to {save_path}")
    
    @staticmethod
    def load_embeddings(load_path: Path) -> np.ndarray:
        """Load embeddings from disk"""
        with open(load_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f" Loaded embeddings: {embeddings.shape}")
        return embeddings
