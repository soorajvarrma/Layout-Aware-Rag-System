"""
Main ingestion pipeline
"""

from pathlib import Path
from typing import List
import json

from src.config import BERT_PAPER_PATH, PROCESSED_DATA_DIR
from src.ingestion.pdf_parser import LayoutAwarePDFParser
from src.ingestion.chunker import SemanticChunker, SemanticChunk


class IngestionPipeline:
    """Complete ingestion pipeline"""
    
    def __init__(self, pdf_path: Path = BERT_PAPER_PATH):
        self.pdf_path = pdf_path
        self.parser = LayoutAwarePDFParser(pdf_path)
        self.chunker = SemanticChunker(chunk_size=512, overlap=50)
    
    def run(self) -> List[SemanticChunk]:
        """Execute full ingestion pipeline"""
        print(" Starting ingestion pipeline...")
        
        # Step 1: Parse PDF with layout awareness
        print("\n Step 1: Parsing PDF...")
        doc_chunks = self.parser.parse()
        
        # Save raw chunks
        raw_chunks_path = PROCESSED_DATA_DIR / "raw_chunks.json"
        self.parser.save_chunks(raw_chunks_path)
        
        # Step 2: Create semantic chunks
        print("\n Step 2: Creating semantic chunks...")
        semantic_chunks = self.chunker.chunk_documents(doc_chunks)
        
        # Save semantic chunks
        semantic_chunks_path = PROCESSED_DATA_DIR / "semantic_chunks.json"
        self._save_semantic_chunks(semantic_chunks, semantic_chunks_path)
        
        print(f"\n Ingestion complete!")
        print(f"   - Raw chunks: {len(doc_chunks)}")
        print(f"   - Semantic chunks: {len(semantic_chunks)}")
        
        return semantic_chunks
    
    def _save_semantic_chunks(self, chunks: List[SemanticChunk], path: Path):
        """Save semantic chunks to JSON"""
        chunks_dict = [chunk.to_dict() for chunk in chunks]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunks_dict, f, indent=2, ensure_ascii=False)
        
        print(f" Saved {len(chunks_dict)} semantic chunks to {path}")


def load_semantic_chunks(json_path: Path = PROCESSED_DATA_DIR / "semantic_chunks.json") -> List[SemanticChunk]:
    """Load semantic chunks from JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_dict = json.load(f)
    
    return [SemanticChunk(**chunk) for chunk in chunks_dict]
