# agents/runner_agent.py

import os
from simulation_sir import run_sir_model

class RunnerAgent:
    def __init__(self, memory):
        self.memory = memory
        
    def run(self, params, output_path="logs"):
        os.makedirs(output_path, exist_ok=True)
        result = run_sir_model(params, output_path)
        self.memory.save("last_run_params", params)
        self.memory.save("output_directory", output_path)
        return result
    