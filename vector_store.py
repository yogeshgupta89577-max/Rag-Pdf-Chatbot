"""
vector_store.py — PDF ingestion pipeline.

Flow:
    PDF File
      → PyPDFLoader          (load raw text from each page)
      → RecursiveCharacterTextSplitter  (split into overlapping chunks)
      → FastEmbedEmbeddings  (convert text → vectors, local ONNX model, free)
      → FAISS Vector Store   (store & search vectors in-memory, per session)
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
    PERSIST_VECTOR_STORE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────
_embeddings_cache = None


def get_embeddings() -> FastEmbedEmbeddings:
    """
    Load a local, free, ONNX-based embedding model (no API key, no torch).
    Cached at module level so the model is only initialised once per process.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        print(f"⏳ Loading local embedding model ({EMBEDDING_MODEL})...")
        _embeddings_cache = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        print("✅ Embedding model loaded.")
    return _embeddings_cache


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading & Chunking
# ─────────────────────────────────────────────────────────────────────────────
def load_and_split_pdf(pdf_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    print(f"📄 Loading PDF: {os.path.basename(pdf_path)}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   └─ {len(documents)} pages loaded.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"   └─ {len(chunks)} chunks created "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store — Create / Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def create_vector_store(pdf_path: str) -> FAISS:
    """Full ingestion pipeline: PDF → chunks → embeddings → FAISS index (in-memory)."""
    chunks     = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    print("⚡ Building FAISS vector store…")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    if PERSIST_VECTOR_STORE:
        vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"✅ Vector store saved → {VECTOR_STORE_PATH}")
    else:
        print("✅ Vector store built in-memory (persistence disabled).")

    return vectorstore


def load_vector_store() -> FAISS:
    """Load an existing FAISS index from disk. Only works if PERSIST_VECTOR_STORE=true."""
    if not PERSIST_VECTOR_STORE:
        raise RuntimeError(
            "PERSIST_VECTOR_STORE is disabled — there is nothing on disk to load. "
            "Set PERSIST_VECTOR_STORE=true in .env if you need this."
        )
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"No vector store found at '{VECTOR_STORE_PATH}'. "
            "Call create_vector_store() first."
        )
    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✅ Vector store loaded from {VECTOR_STORE_PATH}")
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────
def get_retriever(vectorstore: FAISS):
    """Wrap the vector store in a LangChain Retriever."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )