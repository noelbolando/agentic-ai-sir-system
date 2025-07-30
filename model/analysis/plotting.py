"""model.analysis.plotting.py"""

# Import libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(log, today, line_width):
    """Plot the S, I, R counts data for the simulation."""
    log_array = np.array(log)
    plot_data = log_array.transpose()
    dates = pd.date_range(start=today, periods=len(plot_data[0]), freq="D")
    susceptible, infected, recovered = plot_data

    plt.figure(figsize=(10, 6))
    plt.plot(dates, susceptible, label="S", color="blue", linewidth=line_width)
    plt.plot(dates, infected, label="I", color="red", linewidth=line_width)
    plt.plot(dates, recovered, label="R", color="green", linewidth=line_width)

    plt.xlabel("Date")
    plt.ylabel("Population")
    plt.title("SIR Simulation Results")
    plt.legend()
    plt.grid()
    plt.show()

def plot_all_runs(logs, today, line_width):
    """Plot all SIR curves from multiple runs."""
    num_runs = len(logs)
    dates = pd.date_range(start=today, periods=logs[0].shape[0], freq="D")

    plt.figure(figsize=(12, 6))

    for run_log in logs:
        log_array = np.array(run_log).T
        susceptible, infected, recovered = log_array
        plt.plot(dates, susceptible, color="blue", alpha=0.2)
        plt.plot(dates, infected, color="red", alpha=0.2)
        plt.plot(dates, recovered, color="green", alpha=0.2)

    # Plot average across all runs
    mean_log = np.mean(logs, axis=0).T
    mean_s, mean_i, mean_r = mean_log
    plt.plot(dates, mean_s, color="blue", linewidth=line_width, label="Avg Susceptible")
    plt.plot(dates, mean_i, color="red", linewidth=line_width, label="Avg Infected")
    plt.plot(dates, mean_r, color="green", linewidth=line_width, label="Avg Recovered")

    plt.xlabel("Date")
    plt.ylabel("Population")
    plt.title(f"SIR Simulation Across {num_runs} Runs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
  
