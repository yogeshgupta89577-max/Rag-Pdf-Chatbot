"""
prompt_templates.py — All LangChain PromptTemplates used in the pipeline.

Why separate file?
    Prompts are the most important part of any RAG system.
    Keeping them in one place makes tuning easy.
"""

from langchain_core.prompts import PromptTemplate


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Q&A PROMPT  (used by ConversationalRetrievalChain)
# ─────────────────────────────────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """You are an intelligent PDF assistant powered by RAG (Retrieval-Augmented Generation).
Your job is to answer questions ONLY based on the provided PDF context.

STRICT RULES:
- Answer ONLY from the given context — never make things up.
- If the answer is not in the context, say exactly:
  "This information is not available in the uploaded PDF."
- Keep a helpful, professional tone.
- ALWAYS use the structured format below — no exceptions.

────────────────────────────────────────────────────
📌 Direct Answer:
[Concise, direct answer to the question — 1 to 2 sentences]

📖 Explanation:
[Detailed explanation with relevant information from the PDF — 2 to 4 sentences]

🔍 Source Reference:
[Which section / page / topic in the PDF this information is from]

💡 Key Takeaway:
[The single most important insight related to this question — 1 sentence]
────────────────────────────────────────────────────

Context from PDF:
{context}

Chat History:
{chat_history}

Human Question: {question}

Your structured response:"""

QA_PROMPT = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SUMMARY MAP PROMPT  (applied to each individual chunk)
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_MAP_TEMPLATE = """Analyze the following section from a PDF document and extract key information:

{text}

Extract concisely:
- Main topic of this section
- Key facts, data, or figures mentioned
- Important concepts introduced
- Any conclusions or recommendations

SECTION ANALYSIS:"""

SUMMARY_MAP_PROMPT = PromptTemplate(
    template=SUMMARY_MAP_TEMPLATE,
    input_variables=["text"],
)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SUMMARY COMBINE PROMPT  (merges all chunk-level analyses)
# ─────────────────────────────────────────────────────────────────────────────
SUMMARY_COMBINE_TEMPLATE = """You are creating a comprehensive summary of a PDF document.
Based on the section analyses below, produce a well-structured final summary.

{text}

Write the summary in this EXACT format:

📄 DOCUMENT OVERVIEW
[2–3 sentences explaining what this document is about and its purpose]

🎯 MAIN TOPICS COVERED
• [Topic 1]
• [Topic 2]
• [Topic 3]
(add more if needed)

📊 KEY FACTS & FIGURES
• [Important statistic or fact 1]
• [Important statistic or fact 2]
(add more if found in document)

💡 CORE CONCEPTS
• [Concept 1]: [Brief explanation]
• [Concept 2]: [Brief explanation]
(add more as needed)

🔑 KEY TAKEAWAYS
1. [Most important insight]
2. [Second most important insight]
3. [Third most important insight]

⚡ TL;DR
[Single sentence capturing the entire document in plain language]

COMPREHENSIVE SUMMARY:"""

SUMMARY_COMBINE_PROMPT = PromptTemplate(
    template=SUMMARY_COMBINE_TEMPLATE,
    input_variables=["text"],
)