# agents.ui_agent.py

import json

from utils.llm_utils import OllamaLLM

class UIAgent:
    def __init__(self, model="mistral", test_mode=False, params_file="utils/params.json"):
        self.llm = OllamaLLM(model)
        self.test_mode = test_mode
        self.params_file = params_file
    
    def get_user_input(self):
        user_input = input("\n [UI Agent]: What would you like to do?\n> ")
        return user_input

    def classify_intent(self, user_input: str) -> str:
        """
        Classifies the intent of the user prompt.
        
        States:
            Run: run the simulation
            Analyze: analyze the results of the simulation

        """
        prompt = f"""
        You are a helpful assistant. A user has entered the following message:

        "{user_input}"

        Determine whether the user is trying to:
        - 'run' a simulation
        - 'analyze' the results
        - or something else (return 'unknown')

        Respond with only one word: 'run', 'analyze', 'exit', or 'unknown'.
        """
        intent = self.llm.generate(prompt).strip().lower()
        if intent in ["run", "analyze"]:
            return intent
        elif "run" in user_input.lower():
            return "run"
        elif "analyze" in user_input.lower():
            return "analyze"
        elif any(keyword in user_input.lower() for keyword in ["assume", "assumptions", "parameters", "how does", "how does the model", "explain the model"]):
            return "assumptions"
        elif "exit" in user_input.lower():
            return "exit"
        return "unknown"

    def ask_analysis_question(self) -> str:
        """Prompt the user for a specific analysis question."""
        return input("\n [UI Agent]: What would you like to analyze?\n> ")

    def follow_up(self, message: str) -> str:
        prompt = f"""
        The following message should be displayed to the user:

        "{message}"

        Write a user-friendly version of this to display back to them.
        """
        return self.llm.generate(prompt)
    
    def load_params(self):
        """Load the parameters from the file if they exist."""
        try:
            with open(self.params_file, 'r') as file:
                params = json.load(file)
            return params
        except FileNotFoundError:
            return None  # No params file exists, return None.

    def save_params(self, params):
        """Save the parameters to the params.json file."""
        with open(self.params_file, 'w') as file:
            json.dump(params, file, indent=4)

    def get_user_params(self, prompt, default_value, value_type):
        """Get user input and validate it."""
        model_params = input(f"{prompt} (default {default_value}): ")
        if not model_params:
            return default_value
        try:
            return value_type(model_params)  # Convert the input to the appropriate type.
        except ValueError:
            print(f"Invalid input, using default: {default_value}")
            return default_value

    def prompt_for_parameters(self, default_params):
        """Prompt the user for parameters, and return validated ones."""
        print("Would you like to use the default parameters or enter new ones?")
        use_defaults = input("Enter 'yes' to use default, or 'no' to enter custom parameters: ").lower()

        if use_defaults == 'yes' or not use_defaults:
            print("Using default parameters.")
            return default_params

        # Prompt for each parameter
        print("Please enter the following parameters:")

        num_runs = self.get_user_params("Number of runs", default_params["num_runs"], int)
        num_agents = self.get_user_params("Number of agents", default_params["num_agents"], int)
        num_steps = self.get_user_params("Number of steps", default_params["num_steps"], int)
        num_contacts = self.get_user_params("Number of contacts per step", default_params["num_contacts"], int)
        infection_prob = self.get_user_params("Infection probability", default_params["infection_prob"], float)
        infection_duration = self.get_user_params("Infection duration (in steps)", default_params["infection_duration"], int)
        recovery_prob = self.get_user_params("Recovery probability", default_params["recovery_prob"], float)

        # Create a dictionary of the parameters
        new_params = {
            "seed": 42,  # Fixed seed
            "num_runs": num_runs,
            "num_agents": num_agents,
            "num_steps": num_steps,
            "num_contacts": num_contacts,
            "infection_prob": infection_prob,
            "infection_duration": infection_duration,
            "recovery_prob": recovery_prob
        }

        # Save the new parameters to the file
        self.save_params(new_params)

        return new_params
