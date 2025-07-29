# simulation_sir.py

import csv
import os
import random

def run_sir_model(params, output_path):
    for run_id in range(params.get("runs", 3)):
        filename = os.path.join(output_path, f"run_{run_id}.csv")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "num_susceptible", "num_infected", "num_recovered"])
            for t in range(params.get("timesteps", 50)):
                writer.writerow([
                    t,
                    random.randint(100, 800),
                    random.randint(0, 500),
                    random.randint(0, 200)
                ])
    return output_path