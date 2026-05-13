# Layout-Aware RAG System for Academic Papers

A high-precision AI backend for semantic retrieval and reasoning over dense academic papers with complex layouts.

## 🎯 Challenge
Process the BERT paper (Devlin et al., 2019) with:
- Multi-column layout handling
- Table extraction
- Figure comprehension
- Stateless high-precision QA

## 🚀 Quick Start


# Clone repository


# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Git Bash on Windows

# Install dependencies
pip install -r requirements.txt

# Set up API key
# Add GROQ_API_KEY to .env file

# Download paper
python download_paper.py

# Run Streamlit app
streamlit run app.py

## 📹 Demo Video
https://drive.google.com/file/d/1ioWjCCx-Y4h-8zXq8fiYTIwyDmS2HHHB/view?usp=sharing

## 📊 Architecture
See STRATEGY.md for detailed architecture decisions.

## 🛠️ Tech Stack
PDF Processing: PyMuPDF, pdfplumber
Embeddings: sentence-transformers (MiniLM)
Vector Store: FAISS
LLM: Groq API
UI: Streamlit


## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific test file
python tests/test_pdf_parser.py
python tests/test_retrieval.py
python tests/test_pipeline.py