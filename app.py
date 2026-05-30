import os
import tempfile
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pdfplumber
import pandas as pd
import google.generativeai as genai

st.set_page_config(
    page_title="FinSight — Financial Report Analyser",
    page_icon="📊",
    layout="wide"
)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
}

/* Hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Custom header banner */
.finsight-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.finsight-header h1 {
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.5px;
}
.finsight-header p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
}
.finsight-badge {
    display: inline-block;
    background: #0f3460;
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #1e3a5f;
    margin-right: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1f2e;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #60a5fa !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}

/* Answer box */
.answer-box {
    background: #1a1f2e;
    border: 1px solid #1e3a5f;
    border-left: 3px solid #60a5fa;
    border-radius: 12px;
    padding: 1.5rem;
    color: #e2e8f0;
    line-height: 1.8;
    font-size: 0.95rem;
    margin: 1rem 0;
}

/* Input box */
.stTextInput input {
    background: #1a1f2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.15) !important;
}

/* Selectbox */
.stSelectbox select, [data-baseweb="select"] {
    background: #1a1f2e !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* Primary button */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 500 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease !important;
}
.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1a1f2e !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background: #141820 !important;
    border: 1px solid #1e3a5f !important;
    border-top: none !important;
}

/* Source cards inside expander */
.source-card {
    background: #1a1f2e;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.source-card .source-tag {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #60a5fa;
    margin-bottom: 0.4rem;
}
.source-card .source-text {
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* Grounding badges */
.ground-good {
    background: #052e16;
    border: 1px solid #166534;
    color: #4ade80;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
}
.ground-warn {
    background: #1c1200;
    border: 1px solid #854d0e;
    color: #fbbf24;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
}
.ground-bad {
    background: #1c0a0a;
    border: 1px solid #991b1b;
    color: #f87171;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #1a1f2e !important;
    border: 1px dashed #1e3a5f !important;
    border-radius: 12px !important;
}

/* Divider */
hr {
    border-color: #1e293b !important;
}

/* Section labels */
.section-label {
    color: #64748b;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Indexed doc pills */
.doc-pill {
    display: inline-block;
    background: #0f3460;
    color: #93c5fd;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid #1e3a5f;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Backend functions ─────────────────────────────────────────────────────────

@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@st.cache_resource
def load_vectorstore():
    embedder = load_embedder()
    return Chroma(
      persist_directory="./chroma_db",
        embedding_function=embedder,
        collection_name="financial_reports"
    )

def extract_from_pdf(pdf_path, company, year):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and len(text.strip()) > 50:
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"company": company, "year": year, "page": page_num + 1, "type": "text", "source": f"{company}_{year}.pdf"}
                ))
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_text = f"[TABLE Page {page_num+1}]\n{df.to_markdown(index=False)}"
                        documents.append(Document(
                            page_content=table_text,
                            metadata={"company": company, "year": year, "page": page_num + 1, "type": "table", "source": f"{company}_{year}.pdf"}
                        ))
                    except Exception:
                        pass
    return documents

def index_uploaded_pdf(uploaded_file, company, year, vectorstore):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    docs = extract_from_pdf(tmp_path, company, year)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    os.unlink(tmp_path)
    return len(chunks)

def get_filters(vectorstore):
    collection = vectorstore._collection
    results = collection.get(include=["metadatas"])
    companies = sorted(set(m.get("company", "") for m in results["metadatas"] if m.get("company")))
    years = sorted(set(m.get("year", "") for m in results["metadatas"] if m.get("year")))
    return companies, years

def answer_question(question, vectorstore, company=None, year=None):
    where = {}
    if company and company != "All":
        where["company"] = company
    if year and year != "All":
        where["year"] = year

    docs = vectorstore.similarity_search(question, k=6, filter=where if where else None)
    if not docs:
        return "No relevant information found.", []

    context = ""
    for i, doc in enumerate(docs):
        m = doc.metadata
        context += f"[{i+1}] {m.get('company','?')} {m.get('year','?')} (page {m.get('page','?')}):\n{doc.page_content}\n\n"

    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context provided.
Always cite sources using [1], [2] etc. after each claim.
If information is insufficient, say so clearly — do not make up numbers.
Be concise and structured. Use bullet points for comparisons.

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text, docs

# ── UI ────────────────────────────────────────────────────────────────────────

vectorstore = load_vectorstore()
companies, years = get_filters(vectorstore)

# Header
st.markdown(f"""
<div class="finsight-header">
    <h1>📊 FinSight</h1>
    <p>AI-powered financial report analysis with cited, grounded answers</p>
    <br>
    <span class="finsight-badge">RAG</span>
    <span class="finsight-badge">Gemini 2.0</span>
    <span class="finsight-badge">ChromaDB</span>
    <span class="finsight-badge">{len(companies)} {'report' if len(companies)==1 else 'reports'} indexed</span>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<p class="section-label">Upload Report</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop a PDF here", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        col1, col2 = st.columns(2)
        company_input = col1.text_input("Company", placeholder="TCS")
        year_input = col2.text_input("Year", placeholder="2023")
        if st.button("Index PDF", type="primary"):
            if not company_input or not year_input:
                st.error("Enter company and year.")
            else:
                with st.spinner("Indexing..."):
                    n = index_uploaded_pdf(uploaded_file, company_input, year_input, vectorstore)
                    st.success(f"✓ {n} chunks indexed")
                    companies, years = get_filters(vectorstore)
                    st.cache_resource.clear()

    st.divider()
    st.markdown('<p class="section-label">Filter Results</p>', unsafe_allow_html=True)
    selected_company = st.selectbox("Company", ["All"] + companies, label_visibility="collapsed")
    selected_year = st.selectbox("Year", ["All"] + years, label_visibility="collapsed")

    st.divider()
    st.markdown('<p class="section-label">Indexed Reports</p>', unsafe_allow_html=True)
    pills = "".join([f'<span class="doc-pill">{c}</span>' for c in companies])
    st.markdown(pills, unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="section-label">Stack</p>', unsafe_allow_html=True)
    st.markdown('<span class="doc-pill">all-MiniLM-L6-v2</span>', unsafe_allow_html=True)
    st.markdown('<span class="doc-pill">Gemini 2.5 Flash</span>', unsafe_allow_html=True)
    st.markdown('<span class="doc-pill">ChromaDB</span>', unsafe_allow_html=True)

# Main
st.markdown('<p class="section-label">Ask a question</p>', unsafe_allow_html=True)

examples = [
    "What was the total revenue?",
    "What are the key business segments?",
    "What risks were mentioned?",
    "Summarize the financial highlights",
    "What was the net profit?",
    "What was the operating margin?",
    "How did the company perform compared to last year?"
]

selected_example = st.selectbox("Examples", [""] + examples, label_visibility="collapsed")
question = st.text_input("Question", value=selected_example, placeholder="e.g. What was the revenue growth year over year?", label_visibility="collapsed")

if st.button("Analyse →", type="primary") and question:
    with st.spinner("Searching documents and generating answer..."):
        answer, source_docs = answer_question(question, vectorstore, selected_company, selected_year)

    total_chunks = len(source_docs)
    table_chunks = sum(1 for d in source_docs if d.metadata.get("type") == "table")
    text_chunks = total_chunks - table_chunks
    unique_pages = len(set(d.metadata.get("page") for d in source_docs))
    companies_used = list(set(d.metadata.get("company", "?") for d in source_docs))
    years_used = sorted(set(d.metadata.get("year", "?") for d in source_docs))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks Retrieved", total_chunks)
    col2.metric("Pages Referenced", unique_pages)
    col3.metric("Table Chunks", table_chunks)
    col4.metric("Text Chunks", text_chunks)

    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    cited = answer.count("[")
    if cited >= 3:
        st.markdown(f'<div class="ground-good">✅ Well grounded — {cited} citations found</div>', unsafe_allow_html=True)
    elif cited >= 1:
        st.markdown(f'<div class="ground-warn">⚠️ Partially grounded — {cited} citation(s) found</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ground-bad">❌ No citations found — verify manually</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander(f"📄 View sources — {total_chunks} chunks retrieved"):
        for i, doc in enumerate(source_docs):
            m = doc.metadata
            chunk_type = "🗃️ Table" if m.get("type") == "table" else "📝 Text"
            st.markdown(f"""
<div class="source-card">
    <div class="source-tag">[{i+1}] {m.get('company','?')} · {m.get('year','?')} · Page {m.get('page','?')} · {chunk_type}</div>
    <div class="source-text">{doc.page_content[:300]}...</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📊 Retrieval details"):
        st.markdown(f"**Companies referenced:** {', '.join(companies_used)}")
        st.markdown(f"**Years referenced:** {', '.join(years_used)}")
        st.markdown(f"**Filter:** Company = `{selected_company}` · Year = `{selected_year}`")
        st.markdown(f"**Embedding model:** `all-MiniLM-L6-v2` (local, zero API cost)")
        st.markdown(f"**LLM:** `gemini-2.5-flash`")
        st.markdown(f"**Vector DB:** `ChromaDB` (persistent on disk)")
