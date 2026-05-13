import requests
from src.config import BERT_PAPER_URL, BERT_PAPER_PATH

def download_bert_paper():
    """Download BERT paper if not exists"""
    if BERT_PAPER_PATH.exists():
        print(f"✅ Paper already exists at {BERT_PAPER_PATH}")
        return
    
    print(f"📥 Downloading BERT paper...")
    response = requests.get(BERT_PAPER_URL)
    response.raise_for_status()
    
    with open(BERT_PAPER_PATH, 'wb') as f:
        f.write(response.content)
    
    print(f"✅ Paper downloaded to {BERT_PAPER_PATH}")

if __name__ == "__main__":
    download_bert_paper()
