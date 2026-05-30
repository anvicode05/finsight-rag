import os
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def extract_from_pdf(pdf_path, company, year):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "company": company,
                        "year": year,
                        "page": page_num + 1,
                        "type": "text",
                        "source": os.path.basename(pdf_path)
                    }
                ))
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_text = f"[TABLE Page {page_num+1}]\n{df.to_markdown(index=False)}"
                        documents.append(Document(
                            page_content=table_text,
                            metadata={
                                "company": company,
                                "year": year,
                                "page": page_num + 1,
                                "type": "table",
                                "source": os.path.basename(pdf_path)
                            }
                        ))
                    except Exception:
                        pass
    return documents

def build_index(data_folder="./data"):
    all_docs = []
    for filename in os.listdir(data_folder):
        if not filename.endswith(".pdf"):
            continue
        parts = filename.replace(".pdf", "").split("_")
        company = parts[0]
        year = parts[1] if len(parts) > 1 else "2022"
        path = os.path.join(data_folder, filename)
        print(f"Processing {filename}...")
        docs = extract_from_pdf(path, company, year)
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
        print(f"  → {len(chunks)} chunks")

    print(f"\nTotal: {len(all_docs)} chunks")
    print("Loading local embedding model (first time downloads ~90MB)...")

    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print("Building vector index (no rate limits!)...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embedder,
        persist_directory="./chroma_db",
        collection_name="financial_reports"
    )
    print("\nDone! Index saved to ./chroma_db")

if __name__ == "__main__":
    build_index()