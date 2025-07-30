"""model.utils.uuid_builder.py"""

# Import libraries
import uuid

def generate_simulation_uuid():
    """
    Generate a random UUID for a simulation run.

    Returns:
        str: A randomly generated UUID as a string.
    """
    return str(uuid.uuid4())
