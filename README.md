# 📊 FinSight — Financial Report Analyser

AI-powered RAG system that lets you upload any financial PDF and ask questions in plain English — with cited, grounded answers.

## Live Demo
[Click here to try it →](https://finsight-financial-rag.streamlit.app/)

## How it works
1. Upload any financial PDF (annual report, balance sheet, earnings)
2. The app extracts text + tables, chunks them, and indexes into ChromaDB
3. Ask any question — the app retrieves the most relevant passages
4. Gemini 2.5 Flash generates a cited answer grounded in the document

##  Architecture
## ⚙️ Tech Stack
| Component | Tool |
|---|---|
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB |
| LLM | Google Gemini 2.5 Flash |
| PDF extraction | pdfplumber |
| UI | Streamlit |
| Language | Python 3.13 |

## 🚀 Run locally
```bash
git clone https://github.com/anvicode05/finsight-rag
cd finsight-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your Gemini API key
export GOOGLE_API_KEY=your_key_here

# Index your PDFs (put them in data/ as COMPANY_YEAR.pdf)
python ingest.py

# Run the app
streamlit run app.py
```

##  Features
-  Live PDF upload — index any financial document instantly
-  Metadata filtering — filter by company and year
-  Citation tracking — every answer cites source, page, and company
-  Retrieval metrics — chunks retrieved, pages referenced, table vs text ratio
-  Grounding indicator — shows whether answer is well-cited or potentially hallucinated
-  Table extraction — understands financial tables, not just prose
