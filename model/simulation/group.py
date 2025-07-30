"""model.simulation.group.py"""

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
