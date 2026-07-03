"""
summarizer.py — Auto-generates a structured PDF summary using map-reduce.

Strategy  (map_reduce):
    MAP    → each chunk gets summarized individually by the LLM
    REDUCE → all chunk-summaries are combined into one final summary
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

from config import GROQ_MODEL, GROQ_API_KEY
from prompt_templates import SUMMARY_MAP_PROMPT, SUMMARY_COMBINE_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_llm():
    """
    Return ChatGroq LLM using the hardcoded model and .env API key.
    """
    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY missing in .env file!")

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.3,
        max_tokens=1024,  # Optimised token limit for Render free tier
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Summary Function
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(pdf_path: str) -> str:
    """
    Load a PDF and return a structured Markdown-formatted summary.
    """
    print("\n📋 Generating summary…")

    # ── 1. Load PDF ─────────────────────────────────────────────────────────
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()

    # ── 2. Split into optimized chunks for stable map_reduce ─────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,       # Reduced to keep within token window limits
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,   # Enforce standard string length to bypass heavy tokenizers
    )
    chunks = splitter.split_documents(documents)

    # ── 3. Strict Render Free Tier Chunks Optimization ───────────────────────
    max_chunks = 3             # Lowered to 3 sections to optimize API bandwidth
    if len(chunks) > max_chunks:
        step   = len(chunks) // max_chunks
        chunks = chunks[::step][:max_chunks]

    print(f"   └─ Summarising {len(chunks)} document sections…")

    # ── 4. Build chain ────────────────────────────────────────────────────────
    llm = get_llm()

    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=SUMMARY_MAP_PROMPT,
        combine_prompt=SUMMARY_COMBINE_PROMPT,
        verbose=False,
    )

    # ── 5. Run & return ──────────────────────────────────────────────────────
    result  = chain.invoke({"input_documents": chunks})
    summary = result["output_text"]

    print("✅ Summary generated!")
    return summary