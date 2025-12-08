# app.py

import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Setup ---
CHROMA_DB_PATH = "chroma_database"  # Directory to store Chroma data

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot with PDF Upload + Chroma")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process PDFs if uploaded
if uploaded_files:
    with st.spinner("Processing uploaded PDFs..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(all_docs)

        # Initialize embedding model
        #embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

        # Store in Chroma
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_PATH
        )
        vector_store.persist()

        st.success(f"✅ Processed and stored {len(chunks)} chunks in Chroma!")

# Load models and Chroma DB
with st.spinner("Loading models and index..."):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)
    llm = Ollama(model="gemma:2b-instruct", base_url="http://localhost:11434")

st.success("✅ Models loaded! Ready for your questions.")

# Chat Interface
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        try:
            # Find relevant chunks
            docs = vector_store.similarity_search(user_query, k=4)

            # Prepare context
            context = "\n\n".join(
                [f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}" for doc in docs]
            )

            # Create prompt
            prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question: {user_query}

Answer:"""

            # Get answer from LLM
            response = llm.invoke(prompt)

            # Display answer
            st.markdown("### 📌 Gemma's Answer:")
            st.write(response)

            # Show context
            with st.expander("📄 Relevant Context"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Doc {i+1} - {doc.metadata.get('source', 'unknown')}**")
                    st.markdown(doc.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error: {e}")
