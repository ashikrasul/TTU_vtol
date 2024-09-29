import os

workspace_path = "/home/sim/simulator"
ros_package = "rraaa"
merged_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../build/.config.yml")
compose_file = os.path.join(workspace_path, "docker/docker-compose.yml")
config_yaml_extension = ".yml"

include_fields = [
    "ego_vehicle",
]