# test.test_reporter_agent.py

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from agents.reporter_agent import ReporterAgent

def test_reporter():
    reporter = ReporterAgent(model="mistral")

    # Simulated analysis result from AnalyzerAgent
    sample_analysis = {
        "avg_total_infected": {"average_total_infected": 37},
        "peak_infection": 52,
        "step_of_peak": 17,
        "std_peak_infection": {"std_dev_peak_infected": 6, "mean_peak_infected": 52},
        "average_duration_infected": 11,
        "outbreak_timing": {
            "decline_step": 20,
            "clearance_step": None
        }
    }

    user_question = "How many agents were infected?"
    summary = reporter.report(user_question, sample_analysis)
    print("\n📢 Reporter Output:")
    print(summary)

if __name__ == "__main__":
    test_reporter()