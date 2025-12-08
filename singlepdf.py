import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load environment variables (optional)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Paths
CHROMA_PATH = "chroma_database"

# Streamlit UI
st.set_page_config(page_title="📚 PDF Q&A Chatbot", layout="centered")
st.title("📚 Ask Questions from Your PDF")
st.markdown("Upload a PDF file and ask questions about its contents.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")
query = st.text_input("💬 Ask a question based on the PDF")

# Vector DB creation
@st.cache_resource
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vectordb

# QA Chain creation
@st.cache_resource
def load_qa_chain(pdf_path):
    vectordb = process_pdf(pdf_path)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="gemma:2b-instruct")
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Handle Q&A
if uploaded_file and query:
    with st.spinner("Thinking... 🤔"):
        # Save uploaded file to disk
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process and answer
        qa_chain = load_qa_chain(temp_path)
        answer = qa_chain.run(query)
        st.success("✅ Answer:")
        st.write(answer)
