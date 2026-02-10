Retrieval-Augmented Generation (RAG) Chatbot

Overview
This repository implements a **Retrieval-Augmented Generation (RAG) chatbot** in Python.  
The system loads user documents (PDFs, text files), creates vector embeddings, stores them in a searchable vector database, and generates context-aware responses using a language model. This enables users to query documents in natural language and receive accurate, evidence-grounded answers.


Features
Document ingestion from PDFs and text files  
Semantic embedding generation  
Vector search using FAISS / Chroma  
RAG-based query handling  
Interactive web interface for chat (Flask / possibly Streamlit)  
Supports local / offline retrieval with external LLM inference

How It Works
1.Document Upload & Processing: 
   Text is extracted and chunked from user-provided documents.

2.Embedding Creation: 
   Text chunks are converted into high-dimensional vectors.

3.Vector Database:  
   Embeddings are indexed into a FAISS / Chroma vector store for fast semantic similarity search.

4.Retrieval / Search: 
   For each user question, the vector store returns the most relevant content.

5.Answer Generation: 
   Retrieved context is passed to an LLM which generates a natural language response.

6.User Interface:  
   A web app or API allows users to interact with the chatbot.

Technologies & Libraries
- Python  
- FAISS  (vector search)  
- LangChain / LLM wrapper  
- Document loaders / splitters  
- LLM inference (Gemma / other models)  
- Flask / web interface
