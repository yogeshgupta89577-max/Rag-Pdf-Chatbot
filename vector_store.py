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
from langchain_community.document_loaders import PyMuPDFLoader
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
    EMBED_BATCH_SIZE,
    MAX_PDF_PAGES,
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
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"   └─ {len(documents)} pages loaded.")

    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned "
            "image-only PDF — OCR would be needed, which this app doesn't support yet."
        )

    if len(documents) > MAX_PDF_PAGES:
        raise ValueError(
            f"This PDF has {len(documents)} pages, which exceeds the "
            f"{MAX_PDF_PAGES}-page limit for this deployment. Large PDFs can "
            f"exhaust the free hosting tier's memory mid-processing. Please "
            f"split the PDF into smaller parts, or raise MAX_PDF_PAGES in "
            f"config.py if you're running on a plan with more RAM."
        )

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
    """
    Full ingestion pipeline: PDF → chunks → embeddings → FAISS index (in-memory).

    Chunks are embedded in small batches rather than all at once. Embedding
    everything in a single call spikes memory/CPU enough to get the process
    killed on free-tier hosting (Render, Streamlit Cloud) once a PDF has more
    than ~30-50 chunks. Batching keeps peak memory roughly constant regardless
    of document size.
    """
    chunks     = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    print(f"⚡ Building FAISS vector store in {total_batches} batch(es) "
          f"of up to {EMBED_BATCH_SIZE} chunks each…")

    vectorstore = None
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        print(f"   └─ Embedding batch {batch_num}/{total_batches} "
              f"({len(batch)} chunks)…")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

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