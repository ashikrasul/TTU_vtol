import os
import yaml
from . import constants

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            dict1[key] = deep_merge(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


def parse_hierarchical_config(base_config_path, base_directory):
    config_stack = [base_config_path]
    merged_config = {}

    while config_stack:
        current_config_path = config_stack.pop(0)  # Process one config at a time

        # Load the current config file
        current_config = load_yaml_file(os.path.join(base_directory, current_config_path))

        # Merge the current config into the merged config
        merged_config = deep_merge(merged_config, current_config)

        # Check if there are more includes and add them to the stack (FIFO)
        includes = current_config.get('includes', [])
        if isinstance(includes, list):
            config_stack = includes + config_stack  # Process includes first

    return merged_config


def write_yaml_file(file_path, config):
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Config successfully written to {file_path}")
