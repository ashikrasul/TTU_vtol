import os

base_dir = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
compose_file = os.path.join(base_dir, "docker/docker-compose.yml")
merged_config_path = os.path.join(base_dir, ".config.yml")


