import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FOLDER = "pdfs"
CHROMA_DB_PATH = "chroma_db"  # Specify a directory to store Chroma data

# 1️⃣ Load all PDFs and add metadata
all_docs = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        file_path = os.path.join(PDF_FOLDER, filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)

print(f"✅ Loaded {len(all_docs)} pages from PDFs in '{PDF_FOLDER}'")

# 2️⃣ Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)
print(f"✅ Split into {len(chunks)} chunks.")

# 3️⃣ Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4️⃣ Create Chroma vector store from chunks and embeddings
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH
)

# 5️⃣ Save Chroma DB (persistence happens automatically, but let's persist explicitly)
vector_store.persist()
print(f"✅ Chroma DB saved at '{CHROMA_DB_PATH}'!")
