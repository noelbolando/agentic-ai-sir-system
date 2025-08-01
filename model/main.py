"""model.main.py"""

# Import librarires
from datetime import date
import numpy as np

from model.simulation.environment import Environment
from model.simulation.model import Model
from model.analysis.plotting import plot_all_runs
from model.analysis.export import export_all_runs_to_csv
from model.analysis.statistics import get_peak_sir_stats
from model.utils.config_loader import load_config
from model.utils.uuid_builder import generate_simulation_uuid


def main():
    # Load uuid for simulation run
    simulation_uuid = generate_simulation_uuid()

    # Load simulation configuration
    config = load_config("model/config.yaml")
    # Extract simulation parameters
    sim_config = config["simulation"]

    sim_uuid = simulation_uuid
    nRuns = 30  # Number of runs (can also be added to the YAML file if needed)
    num_agents = sim_config["num_agents"]
    num_steps = sim_config["num_steps"]
    num_contacts = sim_config["num_contacts"]
    infection_prob = sim_config["infection_prob"]
    infection_duration = sim_config["infection_duration"]
    recovery_prob = sim_config["recovery_prob"]
    seed = sim_config["seed"]

    #all_logs = []

    for run_id in range(nRuns):
        sim_uuid = sim_uuid
        environment = Environment(infection_prob, infection_duration, recovery_prob)
        model = Model(num_agents, num_steps, num_contacts, environment, seed + run_id, run_id)
        model.run()
        #all_logs.append(np.array(model.sir_log))
        print(f"Run {run_id + 1} completed.")
    
    today = date.today()
    #plot_all_runs(all_logs, today=today, line_width=2)
    #export_all_runs_to_csv(all_logs, output_dir="output/sir_runs")
    
    # TODO: export this to csv
    #peak_stats = get_peak_sir_stats(all_logs)
    #save_peak_stats_to_db(peak_stats, db) 

if __name__ == "__main__":
    main()
