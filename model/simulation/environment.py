"""model.simulation.environment.py"""

class Environment:
    """Defines the environment parameters for the simulation."""

    def __init__(self, infection_prob, infection_duration, recovery_prob):
        """
        Initialize the environment.

        Args:
            infection_prob (float): Probability of infection.
            infection_duration (float): Duration of infection.
        """
        self.infection_prob = infection_prob
        self.infection_duration = infection_duration
        self.recovery_prob = recovery_prob

    def update(self, infection_prob_new, infection_duration_new, recovery_prob_new):
        """
        Update the environment parameters.

        Args:
            infection_prob_new (float): New probability of infection.
            infection_duration_new (float): New duration of infection.
        """
        self.infection_prob = infection_prob_new
        self.infection_duration = infection_duration_new
        self.recovery_prob = recovery_prob_new
