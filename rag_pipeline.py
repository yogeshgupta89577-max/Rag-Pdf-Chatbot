"""
rag_pipeline.py — Core Q&A pipeline using LangChain RAG.

Full flow for every question:
    User Question
        → ConversationalRetrievalChain
            → Retriever (FAISS similarity search → top-4 chunks)
            → PromptTemplate (question + context + chat history)
            → LLM (Groq — inbuilt, see config.GROQ_MODEL)
            → Structured Answer

Conversation memory keeps previous Q&A pairs so the user can ask
follow-up questions ("What did you mean by that?", "Expand on point 2", etc.)
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from vector_store import create_vector_store, get_retriever
from summarizer import get_llm
from prompt_templates import QA_PROMPT
from config import GROQ_MODEL


class RAGPipeline:
    """
    Wraps the full Retrieval-Augmented Generation pipeline.

    Model and API key are inbuilt — read from config.py / .env.
    No user input required.

    Usage:
        pipeline = RAGPipeline("my_doc.pdf")
        answer   = pipeline.ask("What is this document about?")
        pipeline.clear_memory()   # reset conversation
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chain    = None
        self.memory   = None

        self._build()

    # ─────────────────────────────────────────────────────────────────────────
    # Private — Build the pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _build(self):
        print("\n🔨 Building RAG Pipeline…")

        # Step 1 — Ingest PDF into FAISS vector store
        print("  [1/4] Processing PDF → Vector Store")
        vectorstore = create_vector_store(self.pdf_path)

        # Step 2 — Create retriever (top-k similarity search)
        print("  [2/4] Setting up Retriever")
        retriever = get_retriever(vectorstore)

        # Step 3 — Load LLM (inbuilt, hardcoded via config.py)
        print(f"  [3/4] Loading LLM  →  {GROQ_MODEL}  (Groq)")
        llm = get_llm()

        # Step 4 — Conversation memory
        #   memory_key   = key name expected in the PromptTemplate
        #   return_messages = True → stores as HumanMessage / AIMessage objects
        #   output_key   = tells memory which chain output to remember
        print("  [4/4] Setting up Conversation Memory")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        # Combine into ConversationalRetrievalChain
        #   combine_docs_chain_kwargs → passes our custom QA_PROMPT
        #   return_source_documents=False → we only need the answer text
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=False,
            verbose=False,
        )

        print("✅ RAG Pipeline ready!\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Public — Ask a question
    # ─────────────────────────────────────────────────────────────────────────
    def ask(self, question: str) -> str:
        """
        Query the PDF using the RAG pipeline.

        The chain automatically:
            • searches FAISS for relevant chunks
            • injects context + chat history into the prompt
            • returns a structured LLM response
        """
        if not self.chain:
            return "❌ Pipeline not initialised. Please upload a PDF first."

        response = self.chain.invoke({"question": question})
        return response.get("answer", "Sorry, I could not generate an answer.")

    # ─────────────────────────────────────────────────────────────────────────
    # Public — Memory helpers
    # ─────────────────────────────────────────────────────────────────────────
    def clear_memory(self):
        """Reset conversation history (start a fresh chat session)."""
        if self.memory:
            self.memory.clear()
            print("🗑️  Conversation memory cleared.")

    def get_chat_history(self) -> list:
        """Return raw LangChain message objects from memory."""
        if self.memory:
            return self.memory.chat_memory.messages
        return []