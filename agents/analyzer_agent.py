# agents.analyzer_agent.py

import pandas as pd
from utils.analysis_tools import (
    calculate_peak_infection, 
    calculate_average_total_infected, 
    calculate_peak_infection_std, 
    plot_state_dynamics,
    calculate_average_infection_time
)

class AnalyzerAgent:
    def __init__(self, state_logs: str, infection_logs: str):
        self.state_logs = state_logs
        self.state_data = pd.read_csv(self.state_logs)

        self.infection_logs = infection_logs
        self.infection_data = pd.read_csv(self.infection_logs)
    
    def analyze(self, question: str) -> dict:
        question = question.lower()
        results = {}

        if "how many" in question or "total infected" in question:
            avg_total = calculate_average_total_infected(self.state_data)
            results["avg_total_infected"] = {"total infected": avg_total}

        if "peak infected" in question or "peak infection" in question:
            peak, step = calculate_peak_infection(self.state_data)
            results["peak_infection"] = {"peak infection ": peak}
            results["step_of_peak"] = {"peak step": step}

        if "standard deviation" in question or "std" in question:
            std_peak = calculate_peak_infection_std(self.state_data)
            results["std_peak_infection"] = {"peak infection standard deviation": std_peak}

        if "plot" in question.lower() or "graph" in question.lower():
            print("Generating state dynamics plot...")
            plot_state_dynamics(self.state_data)
        
        if "how long" in question.lower() or "duration" in question.lower() or "infection time" in question.lower():
            avg_duration = calculate_average_infection_time(self.infection_data)
            results["avg_infection_duration"] = {"average duration": avg_duration}

        return results