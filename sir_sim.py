# sir_sim.py

# === Import Libraries ===
from datetime import date
import pandas as pd
import random

from utils.config_loader import load_config

# === Environment ===
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

# === Agent ===
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

# === Group ===     
class Group:
    """Defines a group of agents in the simulation."""

    def __init__(self, agents, group_id):
        """
        Initialize the group.

        Args:
            agents (list): List of Agent objects in the group.
        """
        self.agents = agents
        self.group_id = group_id

        # Assign group_ids for agents
        for agent in self.agents:
            agent.group_id = group_id

    def update(self, environment, step):
        """
        Update all agents in the group.

        Args:
            environment (Environment): The environment instance.
            step (int): Current simulation step.
        """
        group_states = [agent.dState for agent in self.agents]
        infectious_agents = [agent for agent in self.agents if agent.dState == "I"]

        for agent in self.agents:
            agent.update_state(group_states, environment, infectious_agents, step)

# === Model ===
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

    def _log_agent_states(self, step):
        """
        Log agent states per times step.
        Logging to memory.
        """
        self.memory_agent_state_logs = []

        for agent in self.agents:
            self.memory_agent_state_logs.append({
                "run_id": self.run_id,
                "step": step,
                "agent_id": agent.unique_id,
                "state": agent.dState
            })
            agent_state_df = pd.DataFrame(self.memory_agent_state_logs)
            return agent_state_df

    def _log_infections(self, step):
        """
        Log infection events with infector agents and infected agents per time step
        Logging to memory.
        """
        self.infection_event_logs = []

        for agent in self.agents:
            if agent.dState == "I" and agent.infection_step == step:
                infector_id = agent.infected_by
                infector = next((a for a in self.agents if a.unique_id == infector_id), None)

                infected_group_id = getattr(agent, "group_id", None)
                infector_group_id = getattr(infector, "group_id", None) if infector else None

                for agent in self.agents:
                    if agent.dState == "I" and agent.infection_step == step:
                        infector_id = agent.infected_by
                        infector = next((a for a in self.agents if a.unique_id == infector_id), None)

                        if infector is not None:
                            duration = step - infector.infection_step
                            infector_group_id = getattr(infector, "group_id", None)
                            infected_group_id = getattr(agent, "group_id", None)

                            self.infection_event_logs.append({
                                "run_id": self.run_id,
                                "step": step,
                                "infector_id": infector.unique_id,
                                "infector_group_id": infector_group_id,
                                "infected_id": agent.unique_id,
                                "infected_group_id": infected_group_id,
                                "duration": duration
                            })
                        else:
                            self.infection_event_logs.append({
                                "run_id": self.run_id,
                                "step": step,
                                "infector_id": None,
                                "infector_group_id": None,
                                "infected_id": agent.unique_id,
                                "infected_group_id": getattr(agent, "group_id", None),
                                "duration": None
                            })
        infection_event_df = pd.DataFrame(self.infection_event_logs)
        return infection_event_df
    
    def step(self, step):
        """Run a single simulation step."""
        for group in self.groups:
            group.update(self.environment, step)
        self._log_infections(step)
        self.groups = self._initialize_groups()  # Shuffle groups after each step
        self._log_agent_states(step)
    
    def run(self):
        """Run the simulation and log SIR counts."""
        for step in range(self.num_steps):
            self.step(step)

# === Main ===
def main():
    # Load simulation configuration
    config = load_config("utils/config.yaml")
    # Extract simulation parameters
    sim_config = config["simulation"]

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
    