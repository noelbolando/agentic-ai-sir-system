# src/agents/rag_agent.py

"""
RAG Agent is responsible for answering questions about SIR model assumptions
and infectious disease spread by retrieving relevant passages from the local
knowledge base and generating a response with an LLM.

This implementation replaces the original Argo/Milvus pipeline with a
self-contained approach that works on any host:
  1. Load the local knowledge markdown file.
  2. Split it into sections.
  3. Retrieve the most relevant sections via keyword scoring.
  4. Generate a response using a configurable LLM (Ollama or Groq).
"""

import os
from utils.llm_utils import LLMClient

# Default knowledge base path relative to this file
_KNOWLEDGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "sir_model_information.md"
)


class RAGAgent:
    def __init__(
        self,
        backend="ollama",
        model="mistral",
        base_url=None,
        api_key=None,
        knowledge_path=None,
        top_k=3,
    ):
        """
        Args:
            backend (str): "ollama" or "groq".
            model (str): LLM model name.
            base_url (str): Ollama endpoint URL (ollama only).
            api_key (str): API key (groq only).
            knowledge_path (str): Path to the knowledge markdown file.
            top_k (int): Number of top sections to include in the context.
        """
        self.llm = LLMClient(backend=backend, model=model, base_url=base_url, api_key=api_key)
        self.top_k = top_k
        self.chunks = self._load_chunks(knowledge_path or _KNOWLEDGE_PATH)

        self.system = (
            "You are an expert infectious disease AI agent. "
            "Use the provided context to answer the user's question clearly and concisely, "
            "as if speaking to a general audience. "
            "Your goal is to educate and expand understanding."
        )

    # ------------------------------------------------------------------
    # Knowledge base loading
    # ------------------------------------------------------------------

    def _load_chunks(self, path: str) -> list:
        """Load the knowledge file and split it into sections by '##' headings."""
        try:
            with open(path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return ["Knowledge base file not found."]

        # Split on '##' section headings; keep non-empty sections
        raw_sections = content.split("##")
        chunks = [s.strip() for s in raw_sections if s.strip()]
        return chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> list:
        """Score each chunk by keyword overlap with the question and return top_k chunks."""
        words = set(question.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for w in words if w in chunk_lower)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[: self.top_k]]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Main RAG pipeline: retrieve relevant context, then generate an answer."""
        relevant_chunks = self._retrieve(question)
        context = "\n\n---\n\n".join(relevant_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self.llm.generate(prompt, system=self.system)
