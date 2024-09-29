#! /usr/bin/python3

import argparse
import os
import signal
import sys
import time

from utils import constants
from utils.config import parse_hierarchical_config, write_yaml_file
from utils.logging import set_logger, log
from utils.ros import ROSManager
from utils.docker import DockerManager


def get_args():
    parser = argparse.ArgumentParser(description="RRAAA Simulator for Autonomous Air Taxis.")
    parser.add_argument("--config", "-c", type=str,
                        default='configs/single-static.yml',
                        help="Path to the config file")
    args = parser.parse_args()
    return args


class Test:

    def __init__(self, config_path):
        self.config = parse_hierarchical_config(config_path, constants.include_fields)
        write_yaml_file(constants.merged_config_path, self.config)        
        set_logger(self.config['loglevel'])

        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        self.rosmanager = ROSManager(constants.workspace_path)

    def shutdown(self, signal_number):
        self.rosmanager.shutdown()
        log.info(f"Program terminated with signal {signal_number}")
        sys.exit(0)

    def run(self):
        if self.rosmanager.build_workspace():
            time.sleep(2)
        else:
            log.error("Workspace build failed")

        if self.config['launchfile']:
            self.rosmanager.start_launch_file(constants.ros_package, self.config['launchfile'])


if __name__ == "__main__":

    args = get_args()
    test = Test(args.config)
    test.run()
