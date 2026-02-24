# src/agents/reporter_agent.py

"""
Reporter Agent is responsible for reporting analyzed calculations.
"""

# Import dependencies
from utils.llm_utils import LLMClient, OllamaLLM


class ReporterAgent:
    def __init__(self, model="mistral", backend="ollama", base_url=None, api_key=None):
        """
        Args:
            model (str): LLM model name.
            backend (str): "ollama" or "groq".
            base_url (str): Ollama endpoint (ollama only).
            api_key (str): API key (groq only).
        """
        self.llm = LLMClient(backend=backend, model=model, base_url=base_url, api_key=api_key)

    def report(self, user_question, analysis_results):
        """Use an LLM to summarize the analysis results in a human-readable format."""
        prompt = f"""
        You are a expert in infectious disease modeling and are being asked to generate a report, outlining the results of an SIR model.

        The user asked: "{user_question}"

        Here are the analysis results (in dictionary format):
        {analysis_results}

        Please explain the results clearly and concisely in plain language for a general audience.
        Avoid repeating the dictionary format. Focus on summarizing key insights.
        If any values are missing, simply note that you couldn't determine the results.
        """

        return self.llm.generate(prompt)