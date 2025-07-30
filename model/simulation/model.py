"""model.simulation.model.py"""

# Import libraries
import csv
from datetime import datetime
import logging
import os
import random

from simulation.agent import Agent
from simulation.group import Group

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LOG_FILENAME = "agent_states.csv"
INFECTION_LOG_FILENAME = "infection_events.csv"

class Model:
    """Defines the simulation model."""

    def __init__(
            self, 
            num_agents, 
            num_steps, 
            num_contacts, 
            environment, 
            seed, 
            run_id):
        """
        Initialize the model.

        Args:
            num_agents (int): Number of agents.
            num_steps (int): Number of simulation steps.
            num_contacts (int): Size of groups.
            environment (Environment): The environment instance.
            seed (int): Random seed.
            run_id (int): Unique identifier for the simulation run.
        """
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.num_contacts = num_contacts
        self.environment = environment
        self.seed = seed
        self.run_id = run_id

        if not os.path.exists(LOG_FILENAME):
            with open(LOG_FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["run_id", "step", "agent_id", "state"])
        
        if not os.path.exists(INFECTION_LOG_FILENAME):
            with open(INFECTION_LOG_FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["run_id", "step", "infector_agent_id", "infector_group_id", "infected_agent_id", "infected_group_id", "lapsed_infection_time"])
        
        self.agents = self._initialize_agents()
        self.groups = self._initialize_groups()

    def _initialize_agents(self):
        """Initialize agents with unique IDs and one infectious agent."""
        agents = []
        infected_agent_id = random.randint(0, self.num_agents - 1)
        for i in range(self.num_agents):
            state = "I" if i == infected_agent_id else "S"
            agent = Agent(unique_id=i, dState=state)
            if state == "I":
                agent.infection_step = 0
            agents.append(agent)
        return agents

    def _initialize_groups(self, lambda_param=1/10, min_size=3, max_size=30):
        """
        Initialize groups with sizes drawn from an exponential distribution.

        Args:
            lambda_param (float): Rate parameter (1/mean) for the exponential distribution.
            min_size (int): Minimum group size.
            max_size (int): Maximum group size.

        Returns:
            list: List of Group objects.
        """
        random.shuffle(self.agents)
        agents = self.agents[:]
        groups = []
        group_id = 0

        while agents:
            # Sample group size from exponential and clip it
            group_size = int(random.expovariate(lambda_param))
            group_size = max(min_size, min(group_size, max_size))
            if group_size > len(agents):
                group_size = len(agents)
            group_agents = agents[:group_size]
            # Assign group_id to each agent and collect their ids
            agent_ids = []
            for agent in group_agents:
                agent.group_id = group_id
                agent_ids.append((agent.unique_id, agent.dState))
            # Create group
            group = Group(group_agents, group_id)
            groups.append(group)
            agents = agents[group_size:]
            group_id += 1
        return groups

    def step(self, step):
        """Run a single simulation step."""
        for group in self.groups:
            group.update(self.environment, step)
        self._log_infections(step)
        self.groups = self._initialize_groups()  # Shuffle groups after each step
        self._log_agent_states(step)

    def _log_agent_states(self, step):
        """
        Log the states of all agents at the current step to a csv file.

        Args:
            step (int): The current simulation step.
        """
        with open(LOG_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            for agent in self.agents:
                writer.writerow([self.run_id, step, agent.unique_id, agent.dState])

    def _log_infections(self, step):
        """
        Log newly infected agents and their infectors to a csv file.

        Args:
            step (int): The current simulation step.
        """
        for agent in self.agents:
            if agent.dState == "I" and agent.infection_step == step:
                infector_id = agent.infected_by
                infector = next((a for a in self.agents if a.unique_id == infector_id), None)

                infected_group_id = getattr(agent, "group_id", None)
                infector_group_id = getattr(infector, "group_id", None) if infector else None



                with open(INFECTION_LOG_FILENAME, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for agent in self.agents:
                        if agent.dState == "I" and agent.infection_step == step:
                            infector_id = agent.infected_by
                            infector = next((a for a in self.agents if a.unique_id == infector_id), None)
                    
                            if infector is not None:
                                duration = step - infector.infection_step
                                writer.writerow([self.run_id, step, infector.unique_id, infector_group_id, agent.unique_id, infected_group_id, duration])
                            else:
                                duration = None
  
    def run(self):
        """Run the simulation and log SIR counts."""
        self.sir_log = []
        for step in range(self.num_steps):
            self.step(step)
            # Count S, I, R after each step
            counts = {'S': 0, 'I': 0, 'R': 0}
            for agent in self.agents:
                counts[agent.dState] += 1
            self.sir_log.append([counts['S'], counts['I'], counts['R']])

