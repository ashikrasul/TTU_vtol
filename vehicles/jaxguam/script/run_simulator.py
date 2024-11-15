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

# Temporary config file path for each simulation
temp_config_path = '/tmp/tmp_config.yml'

# Define ideal position and bounds for random initialization
ideal_x, ideal_y, ideal_z = -80, 75, 75
range_offset = 20

x_min, x_max = ideal_x - range_offset, ideal_x + range_offset
y_min, y_max = ideal_y - range_offset, ideal_y + range_offset
z_min, z_max = ideal_z - range_offset, ideal_z + range_offset

# Global variable to track pose z-value
z_value_below_threshold = False

def pose_callback(msg):
    """Callback to check z-value from PoseStamped message."""
    global z_value_below_threshold
    if msg.pose.position.z < 35:
        z_value_below_threshold = True

def run_simulation(init_x, init_y, init_z, timeout=180):
    """Run a single simulation with provided initial conditions and timeout."""
    global z_value_below_threshold

    # Load the existing base configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the configuration with random initial points
    config['ego_vehicle']['location']['x'] = init_x
    config['ego_vehicle']['location']['y'] = init_y
    config['ego_vehicle']['location']['z'] = init_z

    # Save the modified config to a temporary file
    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)
    time.sleep(5)
    # Start the land_vtol.py process
    process = subprocess.Popen(["rosrun", "jaxguam", "land_vtol.py", "--config", temp_config_path])

    start_time = time.time()
    z_value_below_threshold = False  # Reset the threshold flag

    try:
        # Main loop to monitor timeout and z-value
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

    # Ensure process is cleaned up if it completes before timeout or z-value check
    process.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations with adjustable timeout.")
    parser.add_argument('--timeout', type=int, default=180, help="Timeout duration in seconds.")
    args = parser.parse_args()

    rospy.init_node('simulation_monitor', anonymous=True)
    rospy.Subscriber('/jaxguam/pose', PoseStamped, pose_callback)

    try:
        for i in range(50):
            init_x = np.random.uniform(x_min, x_max)
            init_y = np.random.uniform(y_min, y_max)
            init_z = np.random.uniform(z_min, z_max)

            print(f"Starting iteration {i + 1} with initial conditions: x={init_x:.2f}, y={init_y:.2f}, z={init_z:.2f}, timeout={args.timeout}s...")
            run_simulation(init_x, init_y, init_z, args.timeout)
            print(f"Iteration {i + 1}: Moving to next iteration.")
    except KeyboardInterrupt:
        print("Simulation interrupted by user. Exiting...")
