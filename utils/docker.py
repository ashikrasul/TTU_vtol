import subprocess
from loguru import logger as log

# For now we do not use this. This creates a dependency that the
# host OS and container base OS be identical, at minimum to have same glibc versions.

class DockerManager:
    def __init__(self, compose_file):
        self.compose_file = compose_file

    def _run_compose_command(self, command):
        full_command = ['docker', 'compose', '-f', self.compose_file] + command
        try:
            result = subprocess.run(full_command, check=True, text=True)
            if result.returncode == 0:
                log.info(f"Command {' '.join(full_command)} executed successfully.")
        except subprocess.CalledProcessError as e:
            log.error(f"Error executing command: {e}")

    def start_all_services(self):
        log.info("Starting services...")
        self._run_compose_command(['up', '-d'])

    def stop_all_services(self):
        log.warning("This container is also a service that will be stopped")
        log.info("Stopping services...")
        self._run_compose_command(['down'])

    def restart_all_services(self):
        log.warning("This container is also a service that will be restarted")
        log.info("Restarting services...")
        self._run_compose_command(['down'])
        self._run_compose_command(['up', '-d'])

    def start_service(self, service_name):
        log.info(f"Starting service: {service_name}...")
        self._run_compose_command(['up', '-d', service_name])

    def stop_service(self, service_name):
        log.info(f"Stopping service: {service_name}...")
        self._run_compose_command(['stop', service_name])

    def run_command_in_container(self, service_name, command):
        log.info(f"Running command inside {service_name} container: {command}")
        self._run_compose_command(['exec', service_name] + command.split())

    def view_logs(self, service_name=None):
        if service_name:
            log.info(f"Viewing logs for service {service_name}...")
            self._run_compose_command(['logs', service_name])
        else:
            log.info("Viewing logs for all services...")
            self._run_compose_command(['logs'])
