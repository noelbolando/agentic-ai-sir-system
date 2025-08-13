# src/agents/reporter_agent.py

"""
Reporter Agent is responsible for reporting analyzed calculations.
"""

# Import dependencies
from utils.llm_utils import OllamaLLM

class ReporterAgent:
    def __init__(self, model="mistral"):
        self.llm = OllamaLLM(model)

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