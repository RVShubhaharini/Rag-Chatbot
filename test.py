import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import os
import pickle

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = 384  # for all-MiniLM-L6-v2

# File paths for persistence
index_file = "faiss_index.bin"
texts_file = "texts.pkl"
qa_log_file = "qa_log.txt"

# Load or initialize FAISS index and texts
if os.path.exists(index_file) and os.path.exists(texts_file):
    index = faiss.read_index(index_file)
    with open(texts_file, "rb") as f:
        texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(embedding_dim)
    texts = []

# Ensure log file exists
if not os.path.exists(qa_log_file):
    with open(qa_log_file, "w", encoding="utf-8") as f:
        f.write("Question & Answer Log\n")
        f.write("=====================\n\n")

# Functions
def save_faiss_and_texts():
    faiss.write_index(index, index_file)
    with open(texts_file, "wb") as f:
        pickle.dump(texts, f)

def pdf_to_text(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        except Exception as e:
            st.warning(f"Skipping page {page_num + 1}: {e}")
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
    global texts, index
    for pdf_file in pdf_files:
        text = pdf_to_text(pdf_file)
        chunks = split_text(text)
        embeddings = embed_model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        index.add(embeddings)
        texts.extend(chunks)
    save_faiss_and_texts()

def query_faiss(query, top_k=1):
    if index.ntotal == 0 or len(texts) == 0:
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

def log_question_answer(question, answer):
    with open(qa_log_file, "a", encoding="utf-8") as f:
        f.write(f"Q: {question}\n")
        f.write(f"A: {answer}\n")
        f.write("="*40 + "\n")

# Streamlit UI
st.title("PDF Question Answering with Gemma 2B Instruct & FAISS (with Persistence)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Ingest PDFs"):
        with st.spinner("Ingesting and indexing PDFs..."):
            ingest_pdfs(uploaded_files)
        st.success("PDFs ingested and indexed! (Saved for next time too)")

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
            log_question_answer(query, answer)
