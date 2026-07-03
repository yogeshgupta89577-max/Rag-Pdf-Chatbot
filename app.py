"""
app.py — Streamlit frontend for the PDF Intelligence Chatbot.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from config import GROQ_API_KEY, GROQ_MODEL
from rag_pipeline import RAGPipeline
from summarizer import generate_summary

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Intelligence Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts & base ───────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ─────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.hero-sub {
    text-align: center;
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* ── Cards ───────────────────────────────────────────────────────────── */
.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Summary box ─────────────────────────────────────────────────────── */
.summary-card {
    background-color: #ffffff !important;
    color: #111827 !important; 
    border: 1px solid #d1d5db;
    border-left: 5px solid #6366f1;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    white-space: pre-wrap;
    line-height: 1.8;
    font-size: 1rem;
    font-weight: 500;
}
.summary-card *, .summary-card p, .summary-card li, .summary-card span {
    color: #111827 !important;
}

/* ── Chat bubbles ────────────────────────────────────────────────────── */
.chat-user {
    background-color: #ede9fe !important;
    color: #1e1b4b !important; 
    border-left: 4px solid #7c3aed;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
}
.chat-user * {
    color: #1e1b4b !important;
}

.chat-bot {
    background-color: #f0fdf4 !important;
    color: #062f4f !important; 
    border-left: 4px solid #22c55e;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.6rem 0;
    font-size: 0.95rem;
    white-space: pre-wrap;
    line-height: 1.7;
}
.chat-bot * {
    color: #062f4f !important;
}

.chat-label-user { font-weight: 600; color: #7c3aed !important; margin-bottom: 0.3rem; }
.chat-label-bot  { font-weight: 600; color: #16a34a !important; margin-bottom: 0.3rem; }

/* ── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
}

[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] strong, 
[data-testid="stSidebar"] span:not(.badge) { 
    color: #e0e7ff !important; 
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { color: #c7d2fe !important; font-size: 0.85rem; }

/* ── Model info box ──────────────────────────────── */
.model-info-box {
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.model-info-label {
    font-size: 0.72rem;
    color: #a5b4fc;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.model-info-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #e0e7ff;
    font-weight: 500;
}

/* ── Pill badge ─────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.badge-green  { background-color: #dcfce7 !important; color: #15803d !important; }
.badge-purple { background-color: #ede9fe !important; color: #6d28d9 !important; }
.badge-blue   { background-color: #dbeafe !important; color: #1d4ed8 !important; }
.badge-orange { background-color: #ffedd5 !important; color: #c2410c !important; }

[data-testid="stSidebar"] .badge-green  { color: #15803d !important; background-color: #dcfce7 !important; }
[data-testid="stSidebar"] .badge-purple { color: #6d28d9 !important; background-color: #ede9fe !important; }
[data-testid="stSidebar"] .badge-blue   { color: #1d4ed8 !important; background-color: #dbeafe !important; }
[data-testid="stSidebar"] .badge-orange { color: #c2410c !important; background-color: #ffedd5 !important; }
/* ── Divider ─────────────────────────────────────────────────────────── */
.divider { border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "rag_pipeline":   None,
        "chat_history":   [],   
        "summary":        None,
        "pdf_processed":  False,
        "pdf_name":       "",
        "question_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Helper — render one chat message
# ─────────────────────────────────────────────────────────────────────────────
def render_message(role: str, content: str):
    if role == "user":
        st.markdown(
            f'<div class="chat-user">'
            f'<div class="chat-label-user">🧑 You</div>{content}'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bot">'
            f'<div class="chat-label-bot">🤖 Assistant</div>{content}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align:center;font-size:1.3rem;margin-bottom:0.2rem;'>"
            "🧠 PDF Chatbot</h2>"
            "<p style='text-align:center;font-size:0.8rem;color:#a5b4fc;'>"
            "Powered by LangChain + RAG</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='border-color:#4338ca;'>", unsafe_allow_html=True)

        # ── Active Model Info ────────────────────────────────────────────────
        st.markdown("**⚡ Active Model**")
        st.markdown(
            f"""
            <div class="model-info-box">
                <div class="model-info-label">Provider</div>
                <div class="model-info-value">⚡ Groq</div>
            </div>
            <div class="model-info-box">
                <div class="model-info-label">Model</div>
                <div class="model-info-value">🦙 {GROQ_MODEL}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Key validation status indicator
        if GROQ_API_KEY:
            st.markdown(
                '<br><span class="badge badge-green">🔑 API Key: Configured ✓</span>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ GROQ_API_KEY not found in .env!", icon="⚠️")

        st.markdown("<hr style='border-color:#4338ca;'>", unsafe_allow_html=True)

        # ── Session stats ────────────────────────────────────────────────────
        st.markdown("**📊 Session Stats**")
        col1, col2 = st.columns(2)
        col1.metric("💬 Questions", st.session_state.question_count)
        col2.metric("📄 PDF",
                    "✅ Ready" if st.session_state.pdf_processed else "⬜ None")

        if st.session_state.pdf_processed:
            st.markdown(
                f'<span class="badge badge-green">📄 {st.session_state.pdf_name[:22]}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr style='border-color:#4338ca;'>", unsafe_allow_html=True)

        # ── Actions ──────────────────────────────────────────────────────────
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.chat_history   = []
            st.session_state.question_count = 0
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.clear_memory()
            st.success("Chat cleared!")
            st.rerun()

        if st.button("🔄  Reset All", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("<hr style='border-color:#4338ca;'>", unsafe_allow_html=True)

        # ── Tech stack ───────────────────────────────────────────────────────
        st.markdown("**🛠️ Tech Stack**")
        for badge in [
            ("🦜 LangChain", "badge-purple"),
            ("⚡ Groq LLM API", "badge-orange"),
            (f"🦙 {GROQ_MODEL}", "badge-green"),
            ("🗄️ FAISS Vector Store", "badge-blue"),
            ("🧬 FastEmbed (local)", "badge-green"),
            ("📄 PyPDF Loader", "badge-purple"),
            ("🎈 Streamlit UI", "badge-blue"),
        ]:
            st.markdown(
                f'<span class="badge {badge[1]}">{badge[0]}</span>&nbsp;',
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Upload & Summarize
# ─────────────────────────────────────────────────────────────────────────────
def tab_upload():
    st.markdown("### 📤 Upload Your PDF")

    # Guard: API key must be present before allowing uploads
    if not GROQ_API_KEY:
        st.error(
            "⚠️  **API Key Missing!** "
            "Make sure **GROQ_API_KEY** is set in your `.env` file and restart the app."
        )
        return

    uploaded_file = st.file_uploader(
        "Drop a PDF here, or click to browse",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        c1, c2, c3 = st.columns(3)
        name = uploaded_file.name
        c1.metric("📄 File", name[:22] + "…" if len(name) > 22 else name)
        c2.metric("📦 Size", f"{uploaded_file.size / 1024:.1f} KB")
        c3.metric("📁 Format", "PDF")

        if st.button("🚀  Process PDF & Generate Summary", type="primary"):
            with st.spinner("🔄  Processing PDF — embedding & indexing…"):
                try:
                    # Save to a temp file (PyPDFLoader needs a path)
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    progress = st.progress(0, text="Loading PDF…")

                    # Build RAG pipeline (loads + embeds + stores in FAISS)
                    pipeline = RAGPipeline(pdf_path=tmp_path)
                    st.session_state.rag_pipeline = pipeline
                    progress.progress(55, text="Generating summary…")

                    # Auto-summary
                    summary = generate_summary(tmp_path)
                    st.session_state.summary       = summary
                    st.session_state.pdf_name      = uploaded_file.name
                    st.session_state.pdf_processed = True
                    progress.progress(100, text="Done!")

                    os.unlink(tmp_path)   # clean up temp file
                    st.success("✅  PDF processed — switch to the Chat tab!")
                    st.balloons()

                except Exception as exc:
                    st.error(f"❌  Error: {exc}")

    # ── Display Summary ──────────────────────────────────────────────────────
    if st.session_state.summary:
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Auto-Generated Summary")
        st.markdown(
            f'<div class="summary-card">{st.session_state.summary}</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Chat
# ─────────────────────────────────────────────────────────────────────────────
def tab_chat():
    if not st.session_state.pdf_processed:
        st.info(
            "👆  Please upload and process a PDF first in the **Upload & Summarize** tab!"
        )
        return

    st.markdown(
        f"### 💬 Chat with **{st.session_state.pdf_name}**",
    )

    # ── Chat history ─────────────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center;padding:2.5rem 0;color:#9ca3af;'>"
            "<h3>👋 Ask anything about your PDF!</h3>"
            "<p>I'll retrieve the most relevant passages and give you a structured answer.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat_history:
            render_message(msg["role"], msg["content"])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Suggested questions ───────────────────────────────────────────────────
    st.markdown("**💡 Quick Questions:**")
    suggestions = [
        "What is the main topic?",
        "Summarize the key points",
        "What are the conclusions?",
        "List all important facts",
    ]
    cols = st.columns(4)
    for i, q in enumerate(suggestions):
        with cols[i]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                _send_question(q)
                st.rerun()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            user_q = st.text_input(
                "Question",
                placeholder="e.g., What are the main findings?",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.form_submit_button("📨 Send")

    if send and user_q.strip():
        _send_question(user_q.strip())
        st.rerun()


def _send_question(question: str):
    """Send a question through the RAG pipeline and store the result."""
    with st.spinner("🔍  Searching PDF & generating answer…"):
        try:
            answer = st.session_state.rag_pipeline.ask(question)
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )
            st.session_state.question_count += 1
        except Exception as exc:
            st.error(f"❌  Error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Architecture
# ─────────────────────────────────────────────────────────────────────────────
def tab_architecture():
    st.markdown("### 🏗️ RAG Pipeline Architecture")

    st.markdown(
        f"""
╔══════════════════════════════════════════════════════════════════════╗
║                     PDF RAG PIPELINE                                ║
╚══════════════════════════════════════════════════════════════════════╝

📄 PDF File Upload
│
▼
┌─────────────────────┐
│   PyPDFLoader       │  ← Extracts text from every page
└──────────┬──────────┘
│
▼
┌────────────────────────────────┐
│  RecursiveCharacterTextSplitter│  ← chunk_size=1000, overlap=200
└──────────┬─────────────────────┘     splits at paragraphs → sentences
│
▼
┌─────────────────────────────┐
│   FastEmbed Embeddings      │  ← BAAI/bge-small-en-v1.5 (local, ONNX)
│   text → vector             │  no API key, no torch needed
└──────────┬──────────────────┘
│
▼
┌─────────────────────┐
│  FAISS Vector Store │  ← Stores all chunk vectors on disk
└──────────┬──────────┘    lightning-fast similarity search
│
═══════════════════════════════════════
QUERY TIME (per question)
═══════════════════════════════════════
│
❓ User Question
│
▼
┌─────────────────────┐
│  Similarity Search  │  ← cosine similarity → top-4 chunks retrieved
└──────────┬──────────┘
│
▼
┌──────────────────────────────────────┐
│  PromptTemplate                      │
│  {{context}} + {{chat_history}} + {{q}}    │  ← context-aware prompt
└──────────┬───────────────────────────┘
│
▼
┌──────────────────────────────────────────────────┐
│  Groq LLM  →  {GROQ_MODEL} (inbuilt)             │
└──────────┬───────────────────────────────────────┘
│
▼
📊 Structured Output
📌 Direct Answer
📖 Explanation
🔍 Source Reference
💡 Key Takeaway

┌─────────────────────────────┐
│  ConversationBufferMemory   │  ← stores Q&A history for follow-ups
└─────────────────────────────┘

"""
    )

    st.markdown("### 🧩 LangChain Components Used")
    components = {
        "DocumentLoaders":  "PyPDFLoader — loads PDF pages as LangChain Document objects",
        "TextSplitters":    "RecursiveCharacterTextSplitter — splits docs into overlapping chunks",
        "Embeddings":       "FastEmbedEmbeddings — local ONNX model, converts text to vectors (free, no API key)",
        "VectorStores":     "FAISS — stores vectors for fast cosine-similarity search",
        "Retrievers":       "VectorStoreRetriever — finds top-k most relevant chunks",
        "PromptTemplates":  "PromptTemplate — injects context + history into LLM prompt",
        "Chains":           "ConversationalRetrievalChain — end-to-end RAG + memory chain",
        "Summarize":        "load_summarize_chain (map_reduce) — summarises long documents",
        "Memory":           "ConversationBufferMemory — keeps full chat history in RAM",
        "LLM":              f"ChatGroq → {GROQ_MODEL} (hardcoded, inbuilt)",
    }
    for name, desc in components.items():
        st.markdown(
            f'<div class="card">'
            f'<span class="badge badge-purple">{name}</span>&nbsp;&nbsp;{desc}'
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_session()

    # Header
    st.markdown(
        '<h1 class="hero-title">🧠 PDF Intelligence Chatbot</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">'
        "Upload any PDF → Auto Summary → Ask Questions → Get Structured Answers"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar (model info + stats only)
    render_sidebar()

    # Tabs
    t1, t2, t3 = st.tabs(
        ["📤 Upload & Summarize", "💬 Chat with PDF", "🏗️ Architecture"]
    )

    with t1:
        tab_upload()

    with t2:
        tab_chat()

    with t3:
        tab_architecture()


if __name__ == "__main__":
    main()