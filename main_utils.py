import os
import json

def load_config(config_name:str="config.json", config_dir:str = "utils"):
    """
    Load configuration from a JSON file.

    Args:
        config_name (str): Path to the JSON config file. Default is 'config.json'.
    
    Returns:
        dict: Configuration data loaded from the JSON file.
    """
    config_path = os.path.join(config_dir, config_name)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} does not exist.")

    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    return config