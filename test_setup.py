#!/usr/bin/env python3
"""Test if environment is set up correctly"""

def test_imports():
    print("🧪 Testing imports...")
    
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF")
    except ImportError as e:
        print(f"❌ PyMuPDF: {e}")
    
    try:
        import pdfplumber
        print("✅ pdfplumber")
    except ImportError as e:
        print(f"❌ pdfplumber: {e}")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers")
    except ImportError as e:
        print(f"❌ sentence-transformers: {e}")
    
    try:
        import faiss
        print("✅ FAISS")
    except ImportError as e:
        print(f"❌ FAISS: {e}")
    
    try:
        from groq import Groq
        print("✅ Groq")
    except ImportError as e:
        print(f"❌ Groq: {e}")
    
    try:
        import streamlit
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")

def test_api_key():
    print("\n🔑 Testing API key...")
    from src.config import GROQ_API_KEY
    
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        print(f"✅ Groq API key loaded: {GROQ_API_KEY[:10]}...")
    else:
        print("⚠️  Groq API key not set properly in .env")

def test_paths():
    print("\n📁 Testing paths...")
    from src.config import BERT_PAPER_PATH, VECTOR_STORE_DIR
    
    if BERT_PAPER_PATH.exists():
        print(f"✅ BERT paper found: {BERT_PAPER_PATH}")
    else:
        print(f"⚠️  BERT paper not found. Run: python download_paper.py")
    
    print(f"✅ Vector store directory: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    test_imports()
    test_api_key()
    test_paths()
    print("\n🎉 Environment setup complete!")
