#!/usr/bin/env python3

import os
import time
import yaml
import rospy
import numpy as np
import subprocess
import argparse
from geometry_msgs.msg import PoseStamped
from utils import constants

# Base config file path (from node_vehicle.py)
config_path = constants.merged_config_path
temp_config_path = '/tmp/tmp_config.yml'

ideal_x, ideal_y, ideal_z = -80, 75, 75
range_offset = 20

x_min, x_max = ideal_x - range_offset, ideal_x + range_offset
y_min, y_max = ideal_y - range_offset, ideal_y + range_offset
z_min, z_max = ideal_z - range_offset, ideal_z + range_offset

z_value_below_threshold = False

def pose_callback(msg):
    """Callback to check z-value from PoseStamped message."""
    global z_value_below_threshold
    if msg.pose.position.z < 35:
        z_value_below_threshold = True

def run_simulation(init_x, init_y, init_z, timeout=180):
    """Run a single simulation with provided initial conditions and timeout."""
    global z_value_below_threshold

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['ego_vehicle']['location']['x'] = init_x
    config['ego_vehicle']['location']['y'] = init_y
    config['ego_vehicle']['location']['z'] = init_z

    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    process = subprocess.Popen(
        ["rosrun", "jaxguam", "land_vtol.py", "--config", temp_config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    start_time = time.time()
    z_value_below_threshold = False

    try:
        while process.poll() is None:
            elapsed_time = time.time() - start_time

            if z_value_below_threshold:
                print("Z-value dropped below 35, terminating process.")
                process.terminate()
                process.wait()
                return

            if elapsed_time > timeout:
                print(f"Timeout reached after {timeout}s, terminating process.")
                process.terminate()
                process.wait()
                return

            rospy.sleep(0.1)

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating process...")
        process.terminate()
        process.wait()
        raise  # Re-raise to exit the program

    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations with adjustable timeout.")
    parser.add_argument('--timeout', type=int, default=50, help="Timeout duration in seconds.")
    args = parser.parse_args()

    rospy.init_node('simulation_monitor', anonymous=True)
    rospy.Subscriber('/jaxguam/pose', PoseStamped, pose_callback)

    try:
        for i in range(5):
            init_x = np.random.uniform(x_min, x_max)
            init_y = np.random.uniform(y_min, y_max)
            init_z = np.random.uniform(z_min, z_max)

            print(f"Starting iteration {i + 1} with initial conditions: x={init_x:.2f}, y={init_y:.2f}, z={init_z:.2f}, timeout={args.timeout}s...")
            run_simulation(init_x, init_y, init_z, args.timeout)
            print(f"Iteration {i + 1}: Moving to next iteration.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Cleaning up and exiting...")
        rospy.signal_shutdown("KeyboardInterrupt")
        os._exit(0)  # Ensure all threads and processes are terminated
