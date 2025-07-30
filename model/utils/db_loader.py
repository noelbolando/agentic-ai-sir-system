"""model.utils.db_loader.py"""

# Import libraries
import psycopg2
import yaml

class Database:
    """Handles database interactions."""

    def __init__(self, secrets):
        """Initialize the database connection."""
        try:
            print("Connecting to the database...")
            self.conn = psycopg2.connect(
                host=secrets["host"],
                port=secrets["port"],
                user=secrets["user"],
                password=secrets["password"],
                dbname=secrets["dbname"]
            )
            print("Database connection established.")
            self.cursor = self.conn.cursor()  # Initialize the cursor
            print("Cursor initialized.")
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            raise

    def load_sim_params(self, sim_config, sim_uuid):
        """
        Load sim_uuid and simulation parameters from a YAML file into the database.
        """
        query = """
            INSERT INTO sim_parameters (
                sim_id, seed, num_agents, num_steps, num_contacts, infection_prob,
                infection_duration, recovery_prob
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        values = (
            sim_uuid,
            sim_config.get('seed'),
            sim_config.get('num_agents'),
            sim_config.get('num_steps'),
            sim_config.get('num_contacts'),
            sim_config.get('infection_prob'),
            sim_config.get('infection_duration'),
            sim_config.get('recovery_prob')
            )
        try:
            self.cursor.execute(query, values)
        except psycopg2.Error as e:
            print(f"Error inserting data into the database: {e}")
            raise

    def load_agent_state_logs(self, run_id, step, agent_id, dState, group_id):
        """
        Insert a single agent state into the database.
        """
        
        query = """
            INSERT INTO agent_state_logs (run_id, step, agent_id, state, group_id)
            VALUES (%s, %s, %s, %s, %s);
        """
        try:
            self.cursor.execute(query, (run_id, step, agent_id, dState, group_id))
        except psycopg2.Error as e:
            print(f"Error inserting data into the database: {e}")
            raise
    
    def load_infection_logs(self, run_id, step, infector_unique_id, infector_group_id, infected_unique_id, infected_group_id, duration):
        """
        Insert a single agent state into the database.
        """
       
        query = """
            INSERT INTO infection_logs (run_id, step, infector_agent_id, infector_group_id, infected_agent_id, infected_group_id, lapsed_infection_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        values = (
            run_id,
            step,
            infector_unique_id,
            infector_group_id,
            infected_unique_id,
            infected_group_id,
            duration
        )
        try:
            self.cursor.execute(query, values)
        except psycopg2.Error as e:
            print(f"Error inserting data into the database: {e}")
            raise
    
    def load_peak_stat(self, run_id, peak_infected, peak_infected_step, peak_recovered, peak_recovered_step, peak_susceptible, peak_susceptible_step):
        """
        Insert a single peak stat record into the database.
        """
        query = """
            INSERT INTO sim_peak_stats (run_id, num_peak_infected, peak_infected_step, num_peak_recovered, peak_recovered_step, num_peak_susceptible, peak_susceptible_step)
            VALUES (%s, %s, %s, %s, %s, %s, %s);
        """
        values = (
            run_id, 
            peak_infected, 
            peak_infected_step, 
            peak_recovered, 
            peak_recovered_step, 
            peak_susceptible, 
            peak_susceptible_step
        )
        try:
            self.cursor.execute(query, values)
        except psycopg2.Error as e:
            print(f"Error inserting peak stat into the database: {e}")
            raise

    def commit(self):
        """Commit the current transaction."""
        try:
            self.conn.commit()
        except psycopg2.Error as e:
            print(f"Error committing transaction: {e}")
            raise

    def close(self):
        """Close the database connection."""
        try:
            self.cursor.close()
            self.conn.close()
        except psycopg2.Error as e:
            print(f"Error closing the database connection: {e}")
            raise
