import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def load_vectorstore():
    embedder = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedder,
        collection_name="financial_reports"
    )

def query(question, vectorstore, company=None, year=None, k=6):
    # Build metadata filter
    where = {}
    if company and company != "All":
        where["company"] = company
    if year and year != "All":
        where["year"] = year

    # Retrieve relevant chunks
    if where:
        docs = vectorstore.similarity_search(question, k=k, filter=where)
    else:
        docs = vectorstore.similarity_search(question, k=k)

    if not docs:
        return "No relevant information found in the documents.", []

    # Build context with citations
    context = ""
    for i, doc in enumerate(docs):
        m = doc.metadata
        context += f"[{i+1}] {m.get('company','?')} {m.get('year','?')} (page {m.get('page','?')}):\n{doc.page_content}\n\n"

    # Build prompt
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context provided below.
Always cite your sources using [1], [2] etc. at the end of each claim.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    # Call Gemini
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text, docs

def get_available_filters(vectorstore):
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])
    companies = sorted(set(m.get("company", "") for m in results["metadatas"]))
    years = sorted(set(m.get("year", "") for m in results["metadatas"]))
    return companies, years