import pandas as pd
from langgraph.graph import StateGraph
from langchain.agents import AgentExecutor
from langchain.agents.structured_chat.base import StructuredChatAgent
from langchain.tools import tool
from langchain.schema import SystemMessage
from typing import TypedDict, Optional
from pydantic import BaseModel
from llm_utils import get_llm
from langchain_community.document_loaders.csv_loader import CSVLoader

data = pd.read_csv("all_agent_logs.csv")
df = data.to_markdown(index=False)

# === Tool input schemas ===
class DummyInput(BaseModel):
    input: str 

# === Define Tools ===
@tool(args_schema=DummyInput)
def describe_data(input: str) -> str:
    """Returns a statistical description of the dataset."""
    return str(df.describe(include="all"))

@tool(args_schema=DummyInput)
def get_column_info(input: str) -> str:
    """Returns column names and types."""
    return str(df.dtypes)

@tool(args_schema=DummyInput)
def summarize_shape(input: str) -> str:
    """Returns the number of rows and columns."""
    return f"{df.shape[0]} rows × {df.shape[1]} columns"

# === LangGraph state ===
class GraphState(TypedDict):
    summary: Optional[str]

# === LangGraph node ===
def run_agent(state: GraphState) -> GraphState:
    tools = [describe_data, get_column_info, summarize_shape]
    llm = get_llm(model="mistral", use_ollama=True)

    system_message = SystemMessage(content="You are a helpful data analyst.")

    agent = StructuredChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
        system_message=system_message
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=False,
        handle_parsing_errors=True
    )

    result = executor.run("Please analyze this dataset and summarize the key insights.")
    return {"summary": result}

# === Build LangGraph workflow ===
workflow = StateGraph(GraphState)

workflow.add_node("analyze", run_agent)
workflow.set_entry_point("analyze")
workflow.set_finish_point("analyze")

graph = workflow.compile()

# === Run it ===
if __name__ == "__main__":
    result = graph.invoke({})
    print("\n--- Summary ---")
    print(result["summary"])
