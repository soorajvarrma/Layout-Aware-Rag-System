"""
Basic end-to-end pipeline tests
"""

import pytest
from src.pipeline_v2 import ImprovedQAPipeline


def test_pipeline_initialization():
    """Test that pipeline initializes correctly"""
    pipeline = ImprovedQAPipeline()
    
    assert pipeline.retriever is not None
    assert pipeline.llm_client is not None
    assert pipeline.generator is not None
    
    print("✓ Pipeline initialized")


def test_pipeline_query():
    """Test basic query functionality"""
    pipeline = ImprovedQAPipeline()
    
    query = "What are the two pre-training tasks used in BERT?"
    response = pipeline.ask(query, top_k=5)
    
    # Check response structure
    assert 'query' in response
    assert 'answer' in response
    assert 'sources' in response
    assert 'num_sources' in response
    
    # Check content
    assert len(response['answer']) > 50, "Answer too short"
    assert response['num_sources'] > 0, "No sources returned"
    
    # Check sources structure
    first_source = response['sources'][0]
    assert 'text' in first_source
    assert 'page' in first_source
    assert 'score' in first_source
    
    print(f"✓ Query processed: {len(response['answer'])} char answer, "
          f"{response['num_sources']} sources")


def test_pipeline_multiple_queries():
    """Test multiple queries don't interfere"""
    pipeline = ImprovedQAPipeline()
    
    queries = [
        "What are the pre-training tasks?",
        "What datasets were used?",
    ]
    
    for query in queries:
        response = pipeline.ask(query, top_k=3)
        assert len(response['answer']) > 0
        assert response['num_sources'] > 0
    
    print(f"✓ Processed {len(queries)} queries successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
