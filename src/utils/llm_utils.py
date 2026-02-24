# utils.llm_utils.py

import requests

try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


class LLMClient:
    """
    Unified LLM client supporting Ollama (local) and Groq (cloud).

    Args:
        backend (str): "ollama" or "groq".
        model (str): Model name (e.g. "mistral" for Ollama, "llama-3.3-70b-versatile" for Groq).
        base_url (str): Ollama API endpoint (ignored for Groq).
        api_key (str): Groq API key (ignored for Ollama).
    """

    def __init__(self, backend="ollama", model="mistral", base_url=None, api_key=None):
        self.backend = backend
        self.model = model
        self.base_url = base_url or "http://localhost:11434/api/generate"
        self.api_key = api_key

    def generate(self, prompt: str, system: str = "") -> str:
        if self.backend == "groq":
            return self._groq_generate(prompt, system)
        return self._ollama_generate(prompt, system)

    def _ollama_generate(self, prompt: str, system: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()

    def _groq_generate(self, prompt: str, system: str) -> str:
        if not _GROQ_AVAILABLE:
            raise ImportError("groq package is not installed. Run: pip install groq")
        client = _GroqClient(api_key=self.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()


class OllamaLLM:
    def __init__(self, model="mistral", base_url="http://localhost:11434/api/generate"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "") -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        response = requests.post(self.base_url, json=payload)
        return response.json()["response"].strip()

def summarize(summary_data: dict) -> str:
    """
    Generates a formatted string summary of useful statistics
    from the provided summary_data dictionary.
    """
    summary = []

    # Peak infection times - handle list or single value gracefully
    peaks = summary_data.get("peak_infection_times")
    if peaks is not None:
        if isinstance(peaks, (list, tuple)):
            peaks_str = ", ".join(str(p) for p in peaks)
        else:
            peaks_str = str(peaks)
        summary.append(f"Peak infection times: {peaks_str}")

    # Average infection duration
    avg_duration = summary_data.get("avg_infection_duration")
    if avg_duration is not None:
        summary.append(f"Average infection duration: {avg_duration:.2f} time steps")

    # Model parameters - dict to key=value pairs string
    params = summary_data.get("model_parameters")
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        summary.append(f"Model parameters: {param_str}")

    # Join all lines into one string separated by newlines
    return "\n".join(summary)

def build_prompt(question: str, summary: str) -> str:
    """
    Builds a prompt that includes the summarized data and the user's question.
    """
    prompt = (
        f"You are an epidemic simulation analysis expert.\n"
        f"Simulation Parameters:\n{summary}\n\n"
        f"Peak Infection Time(s): {summary}\n"
        f"Average Infection Duration: {summary} time steps\n\n"
        f"User Question: {question}\n\n"
        f"Answer in a clear and concise way, referencing the data when helpful."
        f"Only return a portion of the prompt when asked that portion specifically."
    )
    return prompt
