# 🧠 PDF Intelligence Chatbot
### Built with LangChain + RAG + FAISS + Streamlit

Upload any PDF → Get an auto-generated structured summary → Chat with it using RAG.

---

### 🔗 Live Demo
**[Try it here → rag-pdf-chatbot-lncqxb4gkdarnvo93egjdy.streamlit.app](https://rag-pdf-chatbot-lncqxb4gkdarnvo93egjdy.streamlit.app/)**

![PDF Intelligence Chatbot Screenshot](Screenshot.png)

-----

## ✨ Features

| Feature | Details |
|---|---|
| 📤 PDF Upload | Drag & drop any PDF file |
| 📋 Auto Summary | map-reduce summarization via LangChain |
| 💬 RAG Q&A | Retrieves relevant chunks, then answers with LLM |
| 🧠 Structured Output | Every answer has Direct Answer / Explanation / Source / Takeaway |
| 🔁 Memory | Follow-up questions remember previous context |
| 🆓 Free Embeddings | FastEmbed (local ONNX model, no API key, no torch) |
| 🆓 Free LLM | Groq — no cost |

---

## 🏗️ Project Structure

```
pdf_rag_chatbot/
├── app.py                ← Streamlit UI (run this)
├── rag_pipeline.py       ← ConversationalRetrievalChain + Memory
├── summarizer.py         ← load_summarize_chain (map_reduce)
├── vector_store.py       ← PyPDFLoader → TextSplitter → FAISS
├── prompt_templates.py   ← All LangChain PromptTemplates
├── config.py              ← Constants, model names, chunk sizes
├── requirements.txt
├── render.yaml            ← Render Blueprint (one-click deploy)
├── .python-version        ← Python version pin for Render
├── .env.example            ← Copy to .env and add your key
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone / download the project
```bash
cd pdf_rag_chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your API key
```bash
cp .env.example .env
# Open .env and add your Groq key
# Get a FREE key at: https://console.groq.com
```

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser. 🎉

---

## ☁️ Deploying to Render

1. Push this project to a GitHub repo.
2. In Render, create a new **Blueprint** and point it at the repo — `render.yaml` handles the rest.
3. Set `GROQ_API_KEY` in the Render dashboard's environment variables (marked `sync: false` in the blueprint, so it won't be committed to git).
4. Use the **Free** plan for testing. Note the free tier has 512MB RAM — this project is sized to fit comfortably within that (no torch/transformers).

---

## 🔄 How It Works (RAG Pipeline)

```
PDF Upload
   │
   ▼
PyPDFLoader → load all pages as Documents
   │
   ▼
RecursiveCharacterTextSplitter → chunk_size=1000, overlap=200
   │
   ▼
FastEmbedEmbeddings → local ONNX vectors (free, no API key)
   │
   ▼
FAISS Vector Store → in-memory per session (disk persistence opt-in)
   │
   ══════════  Query Time  ══════════
   │
   ❓ User Question
   │
   ▼
Similarity Search → top-4 most relevant chunks
   │
   ▼
PromptTemplate → [context] + [chat_history] + [question]
   │
   ▼
LLM (Groq) → generates structured answer
   │
   ▼
📊 Output:
   📌 Direct Answer
   📖 Explanation
   🔍 Source Reference
   💡 Key Takeaway
```

---

## 🧩 LangChain Components Used

| Component | Module | Purpose |
|---|---|---|
| `PyPDFLoader` | `langchain_community.document_loaders` | Load PDF pages |
| `RecursiveCharacterTextSplitter` | `langchain_text_splitters` | Chunk documents |
| `FastEmbedEmbeddings` | `langchain_community.embeddings` | Create vectors locally |
| `FAISS` | `langchain_community.vectorstores` | Store & search vectors |
| `ConversationalRetrievalChain` | `langchain.chains` | Full RAG chain |
| `ConversationBufferMemory` | `langchain.memory` | Store chat history |
| `load_summarize_chain` | `langchain.chains.summarize` | Map-reduce summary |
| `PromptTemplate` | `langchain_core.prompts` | Custom prompts |
| `ChatGroq` | `langchain_groq` | Groq LLM |

---

## 🔧 Customisation

**Change chunk size** → edit `config.py`
```python
CHUNK_SIZE    = 1000   # increase for more context per chunk
CHUNK_OVERLAP = 200    # increase for better boundary handling
```

**Change number of retrieved chunks** → edit `config.py`
```python
TOP_K_RESULTS = 4      # increase for broader context (may slow down)
```

**Change the answer format** → edit `prompt_templates.py`
Modify `QA_PROMPT_TEMPLATE` to add / remove sections.

**Switch LLM model** → edit `config.py`
```python
GROQ_MODEL = "llama-3.1-8b-instant"   # change to any active Groq model
```

**Enable disk persistence for the vector index** (single-user deployments only) → set in `.env`:
```
PERSIST_VECTOR_STORE=true
```

---

## 📝 Notes

- First run downloads the local embedding model (~50-100MB) — cached after that.
- Large PDFs (100+ pages) may take 1–2 minutes to process.
- By default the FAISS index lives only in memory for that session — nothing is written to disk unless `PERSIST_VECTOR_STORE=true`. This avoids multi-user index-overwrite bugs on shared deployments.
- Conversation memory is in-session only (cleared on page refresh).

---

## 🎓 Learning Outcomes

After building this project you will understand:
- How RAG works end-to-end
- Why we chunk documents and what overlap does
- How embeddings represent meaning as vectors
- How FAISS performs similarity search
- How LangChain chains compose multiple components
- How conversation memory works in multi-turn chatbots
- How map-reduce summarization handles long documents

---

Built as a learning project while studying **LangChain + RAG** 🚀
