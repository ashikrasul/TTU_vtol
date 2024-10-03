import os
import sys
import yaml

from loguru import logger as log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_yaml_file(file_path):
    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    log.info(f"Loading YAML file: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            dict1[key] = deep_merge(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


def parse_includes(config, base_directory):
    for key, value in config.items():
        if isinstance(value, dict):
            if 'include' in value:
                include_file = value['include']
                include_path = os.path.join(base_directory, include_file)
                included_config = load_yaml_file(include_path)
                log.info(f"Including file {include_file} into field {key}")
                config[key] = deep_merge({}, included_config)
            else:
                config[key] = parse_includes(value, base_directory)
    
    return config

def parse_hierarchical_config(config_file):
    log.info(f"Loading config from: {config_file}")
    config = load_yaml_file(config_file)

    # Recursively resolve includes
    return parse_includes(config, os.path.dirname(config_file))


def write_yaml_file(file_path, config):
    base_directory = os.path.dirname(file_path)
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Config successfully written to {file_path}")