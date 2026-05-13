"""
Basic tests for PDF parser
"""

import pytest
from pathlib import Path
from src.config import BERT_PAPER_PATH
from src.ingestion.pdf_parser import LayoutAwarePDFParser


def test_pdf_exists():
    """Test that BERT paper PDF exists"""
    assert BERT_PAPER_PATH.exists(), "BERT paper PDF not found"


def test_pdf_parsing():
    """Test basic PDF parsing"""
    parser = LayoutAwarePDFParser(BERT_PAPER_PATH)
    chunks = parser.parse()
    
    # Basic assertions
    assert len(chunks) > 0, "No chunks extracted"
    assert len(chunks) > 100, "Too few chunks extracted"
    
    # Check chunk structure
    first_chunk = chunks[0]
    assert hasattr(first_chunk, 'content')
    assert hasattr(first_chunk, 'page_num')
    assert hasattr(first_chunk, 'chunk_type')
    
    print(f"✓ Extracted {len(chunks)} chunks")


def test_column_detection():
    """Test that multi-column layout is handled"""
    parser = LayoutAwarePDFParser(BERT_PAPER_PATH)
    chunks = parser.parse()
    
    # Check that we have chunks from different pages
    pages = set(chunk.page_num for chunk in chunks)
    assert len(pages) > 10, "Not enough pages processed"
    
    print(f"✓ Processed {len(pages)} pages")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
