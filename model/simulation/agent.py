"""model.simulation.agent.py"""

# Import libraries
import random

class Agent:
    """Defines an agent in the simulation."""

    def __init__(self, unique_id, dState="S"):
        """
        Initialize the agent.

        Args:
            unique_id (int): Unique identifier for the agent.
            dState (str): Disease state of the agent ('S', 'I', or 'R').
        """
        self.unique_id = unique_id
        self.dState = dState
        self.infected_by = None     # agent_id of infectious agent
        self.infection_step = None  # step at which agent becomes infected

    def update_state(self, group_states, environment, infectious_agents, step):
        """
        Update the state of the agent based on group states and environment.

        Args:
            group_states (list): List of disease states in the group.
            environment (Environment): The environment instance.
            infectious_agents (list): Agents currently in state 'I' in the group.
            step(int): Current simulation step.
        """
        infection_prob = environment.infection_prob
        infection_duration = environment.infection_duration
        recovery_prob = environment.recovery_prob
        nI = group_states.count('I')

        if self.dState == 'S':
            if nI > 0 and random.random() < (1 - (1 - infection_prob) ** nI):
                # Pick a random infector from the infectious agents
                infector = random.choice(infectious_agents)
                self.dState = 'I'
                self.infected_by = infector.unique_id
                self.infection_step = step
        
        elif self.dState == 'I':
            if random.random() < recovery_prob:
                self.dState = 'R'

    def infection_duration(self, current_step):
        """Return how long this agent has been infected at the current step."""
        if self.dState == "I" and self.infection_step is not None:
            return current_step - self.infection_step
        return None
