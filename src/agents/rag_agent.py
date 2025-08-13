# src/agents/rag_agent.py

"""
RAG agent is responsible for:
1. retrieving relevant log data and manuals for guidance 
2. combining these into a prompt
3. sending the prompt to an LLM
"""

# Import libraries
import json as std_json
import numpy as np
# Import dependencies
from utils.argo_utils import run_chat, run_embeddings, run_search

class RAGAgent:
    def __init__(
        self,
        collection_name="sir_collections",
        output_fields=["page_content"],
        limit=5
    ):
        self.collection_name = collection_name
        self.output_fields = output_fields
        self.limit = limit

        self.embedding_model = "v3large"
        self.chat_model = "gpt4o"

        self.instructions = """
        You are an expert infectious disease AI agent, tasked with the responsibility of analyzing epidemic simulation data.
        Use the following context to answer the user's question.

        Please answer in a concise and friendly manner, as if you were speaking to a general audience.
        Your goal is to educate and expand understanding, so keep it clear, engaging, and smart.
        """

    def generate_embeddings(self, text: str):
        """Generate embeddings using Argo API."""

        print(f"[DEBUG] Generating embeddings for input: {text}")
        embeddings = run_embeddings(model=self.embedding_model, prompts=[text])

        # Normalize expected structure
        if isinstance(embeddings, dict) and "embedding" in embeddings:
            vec = embeddings["embedding"]
            print(f"[DEBUG] Received vector (dict): dim={len(vec[0])}")
            return vec[0]

        elif isinstance(embeddings, list) and isinstance(embeddings[0], list):
            print(f"[DEBUG] Received vector (list): dim={len(embeddings[0])}")
            return embeddings[0]

        raise ValueError(f"[ERROR] Unexpected embedding format: {embeddings}")

    def search_documents(self, vector):
        """Search for relevant documents in Milvus using Argo API."""
        vector = np.array(vector, dtype=np.float32).tolist()
        print(f"[DEBUG] Searching Milvus with vector of dim: {len(vector)}")

        results = run_search(
            collection=self.collection_name,
            data=vector,
            output_fields=self.output_fields,
            limit=self.limit
        )

        if results and "data" in results:
            return [r.get("text_content") for r in results["data"] if r.get("text_content")]
            print(f"[DEBUG] Retrieved {len(docs)} document(s)")
            return docs
        else:
            raise ValueError(f"[ERROR] Failed to retrieve documents. Got: {results}")

    def generate_response(self, context: str, question: str):
        """Generate natural language answer using Argo chat API."""
        prompt = f"Context:\n{context}\n\nQuestion:\n{question}"
        print(f"[DEBUG] Sending prompt to chat model...")

        response = run_chat(
            instructions=self.instructions,
            model=self.chat_model,
            prompt=prompt
        )

        if response:
            print(f"[DEBUG] Chat model response received")
            return response.strip()
        else:
            raise ValueError("[ERROR] Failed to generate response.")

    def answer(self, question: str) -> str:
        """Main RAG pipeline."""
        print(f"[RAG] Answering: {question}")

        # Step 1: Embed question
        question_embeddings = self.generate_embeddings(question)

        # Step 2: Retrieve relevant chunks
        retrieved_docs = self.search_documents(question_embeddings)

        # Step 3: Build context
        context = "\n---\n".join(retrieved_docs)

        # Step 4: Generate and return answer
        return self.generate_response(context=context, question=question)