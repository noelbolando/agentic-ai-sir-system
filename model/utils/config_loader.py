"""model.utils.config_loader.py"""

# Import libraries
import csv
from datetime import datetime
import yaml


def load_config(file_path):
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration data.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    

def log_model_params(yaml_dict, section_key, filename=None):
    """
    Log the model parameters (config.yaml) to the database.

    Args:
        yaml_dict (dict): The loaded YAML config.
        section_key (str): The keyword in the YAML dict.
    """
    params = yaml_dict.get(section_key, {}).copy()  # copy so original not modified
    params["run_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if filename is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{section_key}_parameters_{timestamp}.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["parameter", "value"])
        for key, value in params.items():
            writer.writerow([key, value])
