import os
import sys
import yaml

from loguru import logger as log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import constants


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


def parse_hierarchical_config(config_file, predefined_fields):
    config_file = os.path.realpath(config_file)
    assert os.path.isfile(config_file), f"Config file path invalid: {config_file}"
    log.info(f"Loading top-level config from: {config_file}")
    config = load_yaml_file(config_file)

    base_directory = os.path.dirname(config_file)
    for field in predefined_fields:
        if field in config:
            subdirectory = os.path.join(base_directory, field)
            lower_level_config_file = os.path.join(subdirectory, f"{config[field]}.yml")
            
            if os.path.exists(lower_level_config_file):
                log.info(f"Loading lower-level config for field '{field}' from: {lower_level_config_file}")
                lower_level_config = load_yaml_file(lower_level_config_file)
                config = deep_merge(config, lower_level_config)
            else:
                log.error(f"Warning: {lower_level_config_file} does not exist.")
    
    log.info("Config parsing completed.")
    return config


def write_yaml_file(file_path, config):
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    print(f"Config successfully written to {file_path}")