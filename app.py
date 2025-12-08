import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to store embeddings and texts
embedding_dim = 384  # for all-MiniLM-L6-v2
index = None
texts = []

def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def ingest_pdfs(pdf_files):
    global index, texts

    all_chunks = []
    for pdf_file in pdf_files:
        text = pdf_to_text(pdf_file)
        chunks = split_text(text)
        all_chunks.extend(chunks)

    # Embed chunks
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Create FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    # Store texts for retrieval
    texts.extend(all_chunks)

def query_faiss(query, top_k=3):
    if index is None or len(texts) == 0:
        return []

    query_emb = embed_model.encode([query])
    query_emb = np.array(query_emb).astype('float32')

    distances, indices = index.search(query_emb, top_k)
    results = []
    for idx in indices[0]:
        if idx < len(texts):
            results.append(texts[idx])
    return results

def generate_answer_with_gemma(prompt):
    process = subprocess.Popen(
        ['ollama', 'run', 'gemma:2b-instruct'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding=None
    )
    stdout, stderr = process.communicate(input=prompt.encode('utf-8'))

    if process.returncode != 0:
        st.error(f"Error running ollama: {stderr.decode('utf-8')}")
        return None
    return stdout.decode('utf-8').strip()

def log_qa(query, answer, log_file="qa_log.txt"):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"Question: {query}\n")
        f.write(f"Answer: {answer}\n")
        f.write("=" * 50 + "\n")

# Streamlit UI
st.title("Multi-PDF QA with Gemma 2B & FAISS")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Ingest PDFs"):
        with st.spinner("Ingesting and indexing PDFs..."):
            ingest_pdfs(uploaded_files)
        st.success("PDFs ingested and indexed!")

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Searching relevant context..."):
        relevant_docs = query_faiss(query)

    if len(relevant_docs) == 0:
        st.warning("No documents found. Please ingest PDFs first.")
    else:
        st.write("### Relevant document snippets:")
        for i, doc in enumerate(relevant_docs):
            st.write(f"{i+1}. {doc[:300]}...")

        context = "\n\n".join(relevant_docs)

        prompt = f"""
You are a helpful assistant. Use the following extracted document snippets to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

        with st.spinner("Generating answer with Gemma 2B..."):
            answer = generate_answer_with_gemma(prompt)

        if answer:
            st.write("### Answer:")
            st.write(answer)
            log_qa(query, answer)
