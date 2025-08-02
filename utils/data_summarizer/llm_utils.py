from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama

def get_llm(model="gpt-4o", use_ollama=False):
    if use_ollama:
        return ChatOllama(model=model)
    else:
        return ChatOpenAI(model_name=model, temperature=0)