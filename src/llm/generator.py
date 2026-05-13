"""
Complete answer generation pipeline with enhanced context building
"""

from typing import List, Tuple, Dict, Any
from src.retrieval.retriever import SemanticRetriever
from src.ingestion.chunker import SemanticChunk
from src.llm.client import LLMClient


class AnswerGenerator:
    """End-to-end QA pipeline"""
    
    def __init__(self, retriever: SemanticRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm_client = llm_client
    
    def generate_answer(self, query: str, top_k: int = 5, expand_context: bool = True) -> Dict[str, Any]:
        """
        Generate answer for a query
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            expand_context: Whether to include neighboring chunks
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\nProcessing query: {query}")
        print("Retrieving relevant context...")
        
        # Retrieve more chunks initially
        initial_k = top_k * 2 if expand_context else top_k
        results = self.retriever.retrieve_with_reranking(query, top_k=initial_k)
        
        # Expand context with neighboring chunks if enabled
        if expand_context:
            results = self._expand_with_neighbors(results, top_k)
        else:
            results = results[:top_k]
        
        # Build context from retrieved chunks
        context = self._build_context(results)
        
        print(f"Retrieved {len(results)} relevant chunks")
        print("Generating answer...")
        
        # Generate answer
        answer = self.llm_client.generate_answer(query, context)
        
        # Prepare response
        response = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': chunk.text,
                    'page': chunk.page_num,
                    'type': chunk.chunk_type,
                    'score': float(score),
                    'chunk_id': chunk.chunk_id
                }
                for chunk, score in results
            ],
            'num_sources': len(results)
        }
        
        return response
    
    def _expand_with_neighbors(self, results: List[Tuple[SemanticChunk, float]], 
                               target_count: int) -> List[Tuple[SemanticChunk, float]]:
        """
        Expand retrieved chunks with neighboring chunks from same page
        This helps capture complete context when headers are retrieved
        """
        expanded = []
        seen_ids = set()
        all_chunks = self.retriever.vector_store.chunks
        
        # Create page-based index for quick lookup
        page_chunks = {}
        for chunk in all_chunks:
            if chunk.page_num not in page_chunks:
                page_chunks[chunk.page_num] = []
            page_chunks[chunk.page_num].append(chunk)
        
        # Sort chunks within each page by chunk_id for ordering
        for page in page_chunks:
            page_chunks[page].sort(key=lambda c: c.chunk_id)
        
        for chunk, score in results:
            if len(expanded) >= target_count:
                break
            
            # Add the main chunk
            if chunk.chunk_id not in seen_ids:
                expanded.append((chunk, score))
                seen_ids.add(chunk.chunk_id)
            
            # Find neighbors on same page
            if chunk.page_num in page_chunks:
                chunks_on_page = page_chunks[chunk.page_num]
                try:
                    idx = chunks_on_page.index(chunk)
                    
                    # Add next chunk if available (likely continuation)
                    if idx + 1 < len(chunks_on_page):
                        next_chunk = chunks_on_page[idx + 1]
                        if next_chunk.chunk_id not in seen_ids and len(expanded) < target_count:
                            # Give neighbor slightly lower score
                            expanded.append((next_chunk, score * 0.9))
                            seen_ids.add(next_chunk.chunk_id)
                    
                    # Add previous chunk if current is very short (likely header)
                    if len(chunk.text) < 100 and idx > 0:
                        prev_chunk = chunks_on_page[idx - 1]
                        if prev_chunk.chunk_id not in seen_ids and len(expanded) < target_count:
                            expanded.append((prev_chunk, score * 0.85))
                            seen_ids.add(prev_chunk.chunk_id)
                            
                except ValueError:
                    pass
        
        # Re-sort by score
        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded[:target_count]
    
    def _build_context(self, results: List[Tuple[SemanticChunk, float]], 
                      max_length: int = 4000) -> str:
        """
        Build context string from retrieved chunks
        Groups chunks by page for better readability
        """
        # Group chunks by page
        page_groups = {}
        for chunk, score in results:
            if chunk.page_num not in page_groups:
                page_groups[chunk.page_num] = []
            page_groups[chunk.page_num].append((chunk, score))
        
        context_parts = []
        current_length = 0
        
        # Sort pages
        for page_num in sorted(page_groups.keys()):
            page_header = f"\n{'='*60}\nPAGE {page_num}\n{'='*60}\n"
            
            if current_length + len(page_header) > max_length:
                break
            
            context_parts.append(page_header)
            current_length += len(page_header)
            
            # Add chunks from this page
            for chunk, score in page_groups[page_num]:
                chunk_text = f"\n{chunk.text}\n"
                
                if current_length + len(chunk_text) > max_length:
                    break
                
                context_parts.append(chunk_text)
                current_length += len(chunk_text)
        
        return "".join(context_parts)
    
    def batch_generate(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries
        """
        responses = []
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'='*60}")
            
            response = self.generate_answer(query, top_k=top_k)
            responses.append(response)
        
        return responses
