"""
config.py — Central configuration for PDF RAG Chatbot
All settings, model names, and constants live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Provider: Groq ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"   # Free via Groq (console.groq.com)

# ── Embeddings (Local, free, ONNX-based via FastEmbed) ──────────────────────
# No API key needed. Downloads a small (~50-100MB) ONNX model on first run,
# cached afterwards. Much lighter than sentence-transformers/torch.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── Text Splitter ─────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between consecutive chunks

# ── Retriever ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS = 4      # number of similar chunks to retrieve per query

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTOR_STORE_PATH = "./faiss_index"

# Disk persistence is OPT-IN. On a shared/multi-user Render instance, every
# upload writing to the same folder on disk causes users to silently
# overwrite each other's index. Keep this False unless you specifically
# need the index to survive a restart for a single-user deployment.
PERSIST_VECTOR_STORE = os.getenv("PERSIST_VECTOR_STORE", "false").lower() == "true"

# ── Resource limits (important on free-tier hosting) ────────────────────────
# Embedding all chunks of a large PDF in a single batch can spike memory/CPU
# enough to get the process killed on free tiers (Render, Streamlit Cloud).
# Processing in smaller batches keeps peak memory much lower.
EMBED_BATCH_SIZE = 25

# Soft cap on page count — very large PDFs are still risky even with batching.
# Raise this only if your hosting plan has more RAM/CPU to spare.
MAX_PDF_PAGES = 150