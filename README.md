# Layout-Aware RAG System for Academic Papers

A high-precision AI backend for semantic retrieval and reasoning over dense academic papers with complex layouts.

## 🎯 Challenge
Process the BERT paper (Devlin et al., 2019) with:
- Multi-column layout handling
- Table extraction
- Figure comprehension
- Stateless high-precision QA

## 🚀 Quick Start

```bash
# Clone repository
git clone <your-repo-url>
cd layout-aware-rag

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
