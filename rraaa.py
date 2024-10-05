#! /usr/bin/python3

import argparse
import atexit
import signal
import sys

from utils import constants
from utils.config import parse_hierarchical_config, write_yaml_file
from utils.docker import ContainerManager
from utils.logging import set_logger, log


def get_args():
    parser = argparse.ArgumentParser(description="RRAAA Simulator for Autonomous Air Taxis.")
    parser.add_argument("config", type=str,
                        help="Path to the config file")
    args = parser.parse_args()
    return args


class Test:

    def __init__(self, config_path):
        self.config = parse_hierarchical_config(config_path)
        write_yaml_file(constants.merged_config_path, self.config)        
        set_logger(self.config['loglevel'])

        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        self.containermanager = ContainerManager(
                                    self.config, constants.compose_file)

    def shutdown(self, signal_number=None):
        if hasattr(self, 'containermanager') and self.containermanager:
            self.containermanager.stop_all()
        if signal_number == None:
            log.info(f"Terminated")
        else:
            log.info(f"Terminated with signal {signal_number}")
        sys.exit(0)

    def run_once(self):
        self.containermanager.start_all()
        self.containermanager.build_all_workspaces()
        self.containermanager.launch_in_all()

    def run(self):
        self.run_once()

if __name__ == "__main__":
    args = get_args()
    test = Test(args.config)
    test.run()
