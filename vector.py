import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_FOLDER = "pdfs"
FAISS_INDEX_PATH = "faiss_index"

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

# 4️⃣ Create FAISS index from chunks and embeddings
vector_store = FAISS.from_documents(chunks, embedding_model)

# 5️⃣ Save FAISS index locally
vector_store.save_local(FAISS_INDEX_PATH)
print(f"✅ FAISS index saved at '{FAISS_INDEX_PATH}'!")
