"""
Semantic chunking with context preservation
"""

from typing import List
from dataclasses import dataclass
from src.ingestion.pdf_parser import DocumentChunk

@dataclass
class SemanticChunk:
    """Chunk optimized for retrieval"""
    text: str
    chunk_id: str
    page_num: int
    chunk_type: str
    metadata: dict
    
    def to_dict(self):
        return {
            'text': self.text,
            'chunk_id': self.chunk_id,
            'page_num': self.page_num,
            'chunk_type': self.chunk_type,
            'metadata': self.metadata
        }


class SemanticChunker:
    """
    Intelligent chunking that preserves semantic meaning
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, doc_chunks: List[DocumentChunk]) -> List[SemanticChunk]:
        """Create semantic chunks from document chunks"""
        semantic_chunks = []
        
        # Group by page and type
        text_chunks = [c for c in doc_chunks if c.chunk_type == 'text']
        table_chunks = [c for c in doc_chunks if c.chunk_type == 'table']
        figure_chunks = [c for c in doc_chunks if c.chunk_type == 'figure_caption']
        
        # Process text with sliding window
        for idx, chunk in enumerate(text_chunks):
            # Split long chunks
            text = chunk.content
            
            if len(text) <= self.chunk_size:
                # Small chunk - keep as is
                semantic_chunks.append(
                    SemanticChunk(
                        text=text,
                        chunk_id=f"text_{chunk.page_num}_{idx}",
                        page_num=chunk.page_num,
                        chunk_type='text',
                        metadata=chunk.metadata
                    )
                )
            else:
                # Large chunk - split with overlap
                sub_chunks = self._split_with_overlap(text)
                for sub_idx, sub_text in enumerate(sub_chunks):
                    semantic_chunks.append(
                        SemanticChunk(
                            text=sub_text,
                            chunk_id=f"text_{chunk.page_num}_{idx}_{sub_idx}",
                            page_num=chunk.page_num,
                            chunk_type='text',
                            metadata=chunk.metadata
                        )
                    )
        
        # Tables - keep whole (they're already structured)
        for idx, chunk in enumerate(table_chunks):
            semantic_chunks.append(
                SemanticChunk(
                    text=chunk.content,
                    chunk_id=f"table_{chunk.page_num}_{idx}",
                    page_num=chunk.page_num,
                    chunk_type='table',
                    metadata=chunk.metadata
                )
            )
        
        # Figure captions
        for idx, chunk in enumerate(figure_chunks):
            semantic_chunks.append(
                SemanticChunk(
                    text=chunk.content,
                    chunk_id=f"figure_{chunk.page_num}_{idx}",
                    page_num=chunk.page_num,
                    chunk_type='figure',
                    metadata=chunk.metadata
                )
            )
        
        print(f" Created {len(semantic_chunks)} semantic chunks")
        return semantic_chunks
    
    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text with sliding window overlap"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            i += (self.chunk_size - self.overlap)
        
        return chunks
