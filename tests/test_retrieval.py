"""
Basic tests for retrieval system
"""

import pytest
from src.retrieval.hybrid_retriever import HybridRetriever
from src.config import VECTOR_STORE_DIR


def test_vector_store_exists():
    """Test that vector store is built"""
    assert VECTOR_STORE_DIR.exists(), "Vector store not found"
    assert (VECTOR_STORE_DIR / "faiss_index.bin").exists()
    assert (VECTOR_STORE_DIR / "chunks.pkl").exists()


def test_retrieval_basic():
    """Test basic retrieval functionality"""
    retriever = HybridRetriever.load_from_disk()
    
    # Simple query
    query = "What are the pre-training tasks in BERT?"
    results = retriever.retrieve(query, top_k=5)
    
    # Check results
    assert len(results) == 5, "Should return 5 results"
    
    # Check result structure
    chunk, score = results[0]
    assert hasattr(chunk, 'text')
    assert hasattr(chunk, 'page_num')
    assert score > 0
    
    print(f"✓ Retrieved {len(results)} chunks")


def test_retrieval_quality():
    """Test retrieval quality on known queries"""
    retriever = HybridRetriever.load_from_disk()
    
    test_cases = [
        {
            'query': 'What are the two pre-training tasks?',
            'expected_keywords': ['task', 'pre-train', 'bert']
        },
        {
            'query': 'What datasets were used for pre-training?',
            'expected_keywords': ['corpus', 'data', 'train']
        },
        {
            'query': 'Compare BERT-BASE and BERT-LARGE',
            'expected_keywords': ['base', 'large', 'parameter']
        }
    ]
    
    for case in test_cases:
        results = retriever.retrieve(case['query'], top_k=5)
        
        # Check that we got results
        assert len(results) > 0, f"No results for query: {case['query']}"
        
        # Check if any keywords present in retrieved chunks
        all_text = ' '.join([chunk.text.lower() for chunk, _ in results])
        found_keywords = [kw for kw in case['expected_keywords'] if kw in all_text]
        
        # At least one keyword should be found
        assert len(found_keywords) > 0, \
            f"No expected keywords found for query: {case['query']}"
        
        print(f"✓ Query '{case['query']}' found keywords: {found_keywords}")
    
    print(f"✓ All {len(test_cases)} quality checks passed")


def test_retrieval_scores():
    """Test that retrieval scores are reasonable"""
    retriever = HybridRetriever.load_from_disk()
    
    query = "What is BERT?"
    results = retriever.retrieve(query, top_k=5)
    
    # Check scores are in reasonable range
    for chunk, score in results:
        assert score > 0, "Score should be positive"
        assert score < 10, "Score seems too high"
    
    # Scores should be in descending order
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"
    
    print(f"✓ Scores range from {scores[-1]:.3f} to {scores[0]:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
