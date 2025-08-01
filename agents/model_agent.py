# agents/model_agent.py

from sir_sim import main
from memory.json_memory import JSONMemory

class ModelAgent:
    def __init__(self, memory):
        self.memory = memory
        
    def run(self):
        main()
    