import os


frequency_low = 0.5

base_dir = os.path.abspath(os.path.dirname(__file__))
compose_file = os.path.join(base_dir, "../docker/docker-compose.yml")
merged_config_path = os.path.join(base_dir, "tmp", "config.yml")

landing_target_reached_file = 'target_reached.txt'
