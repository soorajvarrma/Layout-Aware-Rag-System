"""
Test the ingestion pipeline
"""

from src.ingestion.pipeline import IngestionPipeline
from src.config import BERT_PAPER_PATH

def main():
    print("=" * 60)
    print("TESTING PDF INGESTION PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = IngestionPipeline(BERT_PAPER_PATH)
    
    # Run ingestion
    chunks = pipeline.run()
    
    # Display sample chunks
    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS")
    print("=" * 60)
    
    # Show first 3 text chunks
    text_chunks = [c for c in chunks if c.chunk_type == 'text'][:3]
    for i, chunk in enumerate(text_chunks, 1):
        print(f"\n--- Text Chunk {i} (Page {chunk.page_num}) ---")
        print(chunk.text[:200] + "...")
    
    # Show first table
    table_chunks = [c for c in chunks if c.chunk_type == 'table']
    if table_chunks:
        print(f"\n--- Sample Table (Page {table_chunks[0].page_num}) ---")
        print(table_chunks[0].text[:300] + "...")
    
    # Show statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print(f"Text chunks: {len([c for c in chunks if c.chunk_type == 'text'])}")
    print(f"Table chunks: {len([c for c in chunks if c.chunk_type == 'table'])}")
    print(f"Figure chunks: {len([c for c in chunks if c.chunk_type == 'figure'])}")
    
    # Page distribution
    pages = {}
    for chunk in chunks:
        pages[chunk.page_num] = pages.get(chunk.page_num, 0) + 1
    
    print(f"\nChunks per page:")
    for page in sorted(pages.keys())[:5]:  # First 5 pages
        print(f"  Page {page}: {pages[page]} chunks")

if __name__ == "__main__":
    main()
