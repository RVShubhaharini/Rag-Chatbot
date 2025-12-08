# app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Setup ---
FAISS_INDEX_PATH = "faiss_index"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 RAG Chatbot with Mistral + FAISS")

# Load embeddings and vector store
with st.spinner("Loading models and index..."):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

    llm = Ollama(
        model="mistral:7b-instruct-q4_0",
        base_url="http://localhost:11434"
    )
st.success("✅ Models loaded! Ready for your questions.")

# --- Chat Interface ---
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        try:
            # Search in FAISS
            docs = vector_store.similarity_search(user_query, k=4)
            context = "\n\n".join(
                [f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}" for doc in docs]
            )

            # Create prompt
            prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question: {user_query}

Answer:"""

            # Get response from Ollama (Mistral)
            response = llm.invoke(prompt)

            st.markdown("### 📌 Mistral's Answer:")
            st.write(response)

            # Show relevant docs (optional)
            with st.expander("📄 Relevant Context"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Doc {i+1} - {doc.metadata.get('source', 'unknown')}**")
                    st.markdown(doc.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"Error: {e}")
