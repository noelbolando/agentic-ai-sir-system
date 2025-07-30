"""model.analysis.export.py"""

# Import libraries
import csv
import os

def export_all_runs_to_csv(all_logs, output_dir="output", prefix="sir_run"):
    """Export all SIR logs to individual CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for run_id, log in enumerate(all_logs):
        filename = os.path.join(output_dir, f"{prefix}_{run_id + 1}.csv")
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Step", "Susceptible", "Infected", "Recovered"])
            for step, (s, i, r) in enumerate(log):
                writer.writerow([step, s, i, r])
