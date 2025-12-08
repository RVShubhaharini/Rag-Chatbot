# app.py

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- Setup ---
CHROMA_DB_PATH = "chroma_db"  # Directory where your Chroma DB is stored

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 RAG Chatbot with Gemma + Chroma")

# --- Load models and index ---
with st.spinner("Loading models and index..."):
    # Load the embedding model (Hugging Face)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Load the Chroma vector store
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )

    # Setup Ollama LLM (Mistral or Gemma)
    llm = Ollama(
        model="gemma:2b-instruct",  # Or "mistral:7b-instruct-q4_0"
        base_url="http://localhost:11434"  # Your Ollama server
    )

st.success("✅ Models loaded! Ready for your questions.")

# --- Chat Interface ---
user_query = st.text_input("Ask a question:")

if user_query:
    with st.spinner("Thinking..."):
        try:
            # Step 1: Find relevant chunks
            docs = vector_store.similarity_search(user_query, k=4)

            # Step 2: Prepare context
            context = "\n\n".join(
                [f"From {doc.metadata.get('source', 'unknown')}:\n{doc.page_content}" for doc in docs]
            )

            # Step 3: Create prompt
            prompt = f"""Answer the following question based on the context below.

Context:
{context}

Question: {user_query}

Answer:"""

            # Step 4: Get answer from LLM
            response = llm.invoke(prompt)

            # Step 5: Display response
            st.markdown("### 📌 Gemma's Answer:")
            st.write(response)

            # Step 6: Show context (optional)
            with st.expander("📄 Relevant Context"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Doc {i+1} - {doc.metadata.get('source', 'unknown')}**")
                    st.markdown(doc.page_content)
                    st.markdown("---")

        except Exception as e:
            st.error(f"❌ Error: {e}")
