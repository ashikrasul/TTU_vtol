#! /usr/bin/python3

import argparse
import atexit
import signal
import sys
import time

from utils import constants
from utils.config import parse_hierarchical_config, write_flattened_config, read_shared_tmp_file
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
        write_flattened_config(constants.merged_config_path, self.config)
        set_logger(self.config['loglevel'])

        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

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

    def shutdown_handler(self, signal_number, frame):
        self.shutdown(signal_number)

    def simulation_ended(self, filename):
        try:
            if read_shared_tmp_file(filename) == 'True':
                return True
            return False
        except FileNotFoundError:
            return False

    def reset(self):
        # Write config again to accomodate any changes and clean up prior state.
        write_flattened_config(constants.merged_config_path, self.config)
        self.containermanager.stop_all()
        self.containermanager.start_all()
        self.containermanager.run_all()

    def run_once(self):
        self.containermanager.start_all()
        self.containermanager.build_all_workspaces()
        self.containermanager.run_all()
        self.containermanager.wait_for_all()

    # TODO: capture failure conditions
    def run_iterations(self, iter_count):
        self.containermanager.start_all()
        self.containermanager.build_all_workspaces()

        log.info("Starting Iteration", 1)
        self.containermanager.run_all()

        for iter in range(iter_count):
            while self.simulation_ended(constants.landing_target_reached_file) == False:
                time.sleep(1)
            if iter < iter_count - 1:
                log.info("Starting Iteration", iter + 2)
                self.reset()
            else:
                self.containermanager.stop_all()

    def run(self):
        mode = self.config['simulation_mode']
        if mode == 'simple':
            self.run_once()
        elif isinstance(mode, int):
            self.run_iterations(mode)
        else:
            log.error(f"Unknown simulation mode {mode}")
            sys.exit(-1)

if __name__ == "__main__":
    args = get_args()
    test = Test(args.config)
    test.run()
