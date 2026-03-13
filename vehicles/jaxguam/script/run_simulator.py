#!/usr/local/bin/py310ros
import os
import time
import yaml
import rospy
import numpy as np
import subprocess
import argparse
from threading import Lock
from geometry_msgs.msg import PoseStamped, Vector3
from tf.transformations import euler_from_quaternion
from utils import constants
from loguru import logger
from plot_results import (
    create_run_folder,
    save_results_to_csv,
    save_summary_to_csv_and_metadata,
)

from utils.config import load_yaml_file, write_shared_tmp_file


config_path = constants.merged_config_path
temp_config_path = "/tmp/tmp_config.yml"

ideal_x, ideal_y, ideal_z = -48, 134, 7
# range_offset = [40, 40, 120]
range_offset = [40, 40, 120]

x_min, x_max = ideal_x - range_offset[0], ideal_x + range_offset[0]
y_min, y_max = ideal_y - range_offset[1], ideal_y + range_offset[1]
z_min, z_max = ideal_z + 35, ideal_z + range_offset[2]

initial_positions = []
final_positions = []
landing_times = []
landing_results = []
final_euler_angles = []

# --- Robust init gating parameters ---
INIT_WAIT_TIMEOUT = 20.0     # seconds to wait for pose near requested init pose
FRESH_MSG_MAX_AGE = 0.5      # seconds: pose must be this recent
INIT_TOL_XY = 2.0            # meters: how close pose must be to init_x/init_y
INIT_TOL_Z = 3.0             # meters: how close pose must be to init_z
ZERO_VEL_STOP_SEC = 5
STREAM_CHILD_OUTPUT = True  # set True if you want land_vtol.py output live

target_reached = False
class SimulationMonitor:
    def __init__(self):
        self.lock = Lock()
        self.reset()

        rospy.Subscriber("/jaxguam/pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/controller_node/vel_cmd", Vector3, self.velocity_callback)


    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg.pose.position
            self.last_pose_time = time.time()

            # landing event only valid after episode_started
            if self.episode_started and msg.pose.position.z < ideal_z:
                self.z_value_below_threshold = True
                q = (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                )
                self.final_angles = euler_from_quaternion(q)

    def velocity_callback(self, msg):
        with self.lock:
            self.last_vel_time = time.time()

            if not self.episode_started:
                self.velocity_zero_start_time = None
                self.zero_velocity_duration = 0.0
                return

            if msg.x == 0 and msg.y == 0 and msg.z == 0:
                if self.velocity_zero_start_time is None:
                    self.velocity_zero_start_time = time.time()
                self.zero_velocity_duration = time.time() - self.velocity_zero_start_time
            else:
                self.velocity_zero_start_time = None
                self.zero_velocity_duration = 0.0

    def reset(self):
        with self.lock:
            self.reset_time = time.time()
            self.current_pose = None
            self.final_angles = None

            self.last_pose_time = None
            self.last_vel_time = None

            self.episode_started = False
            self.z_value_below_threshold = False

            self.velocity_zero_start_time = None
            self.zero_velocity_duration = 0.0


def _pose_is_fresh(monitor):
    """True if we have a pose message that arrived recently and after reset()."""
    now = time.time()
    if monitor.last_pose_time is None:
        return False
    if monitor.last_pose_time < monitor.reset_time:
        return False
    return (now - monitor.last_pose_time) <= FRESH_MSG_MAX_AGE


def wait_for_init_pose(monitor, rate, init_x, init_y, init_z, timeout=INIT_WAIT_TIMEOUT):
    """
    Wait until we observe a fresh pose close to the requested init pose.
    This is the key 'foolproof' guard against stale final pose from last run.
    """
    deadline = time.time() + timeout
    while time.time() < deadline and not rospy.is_shutdown():
        with monitor.lock:
            pose = monitor.current_pose
            fresh = _pose_is_fresh(monitor)

        if pose is not None and fresh:
            dx = abs(pose.x - init_x)
            dy = abs(pose.y - init_y)
            dz = abs(pose.z - init_z)

            if dx <= INIT_TOL_XY and dy <= INIT_TOL_XY and dz <= INIT_TOL_Z:
                return True, (dx, dy, dz)

        rate.sleep()

    # return best-effort diagnostics
    with monitor.lock:
        pose = monitor.current_pose
        last_pose_time = monitor.last_pose_time
    return False, (pose, last_pose_time)


def run_simulation(monitor, init_x, init_y, init_z, timeout=150):
    monitor.reset()

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["ego_vehicle"]["location"]["x"] = float(init_x)
    config["ego_vehicle"]["location"]["y"] = float(init_y)
    config["ego_vehicle"]["location"]["z"] = float(init_z)

    with open(temp_config_path, "w") as file:
        yaml.safe_dump(config, file)

    with open(constants.simulation_status_file, "w") as status_file:
        status_file.write("False")

    popen_kwargs = {}
    if STREAM_CHILD_OUTPUT:
        popen_kwargs["stdout"] = None
        popen_kwargs["stderr"] = None
    else:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE

    process = subprocess.Popen(
        ["rosrun", "jaxguam", "land_vtol.py", "--config", temp_config_path],
        **popen_kwargs,
    )

    start_time = time.time()
    stop_reason = None
    landing_pose = None                  # ← ADDED: snapshot at stop moment

    try:
        # ---- FOOLPROOF START GATE ----
        ok, info = wait_for_init_pose(monitor, rate, init_x, init_y, init_z, timeout=INIT_WAIT_TIMEOUT)
        if not ok:
            stop_reason = f"Init failed: never observed pose near init within {INIT_WAIT_TIMEOUT}s. info={info}"
            logger.error(stop_reason)
            return stop_reason, None     # ← CHANGED: return tuple

        dx, dy, dz = info
        with monitor.lock:
            monitor.episode_started = True
        logger.info(f"Init confirmed (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}). Episode started.")
        # -----------------------------

        while process.poll() is None:
            elapsed = time.time() - start_time

            with monitor.lock:
                landed = monitor.z_value_below_threshold
                zero_vel = monitor.zero_velocity_duration

            if landed:
                with monitor.lock:                           # ← ADDED
                    landing_pose = monitor.current_pose  
                    landing_angles = monitor.final_angles    # ← ADDED
                stop_reason = f"Z dropped below {ideal_z}m"
                logger.warning(f"Iteration stopped: {stop_reason}")
                break

            if zero_vel >= ZERO_VEL_STOP_SEC:
                with monitor.lock:                           # ← ADDED
                    landing_pose = monitor.current_pose
                    landing_angles = monitor.final_angles       # ← ADDED
                stop_reason = f"Velocities zero for {ZERO_VEL_STOP_SEC} seconds"
                logger.warning(f"Iteration stopped: {stop_reason}")
                break

            if elapsed > timeout:
                with monitor.lock:                           # ← ADDED
                    landing_pose = monitor.current_pose
                    landing_angles = monitor.final_angles       # ← ADDED
                stop_reason = f"Timeout reached after {timeout}s"
                logger.warning(f"Iteration stopped: {stop_reason}")
                break

            rate.sleep()

    except KeyboardInterrupt:
        stop_reason = "KeyboardInterrupt"
        logger.error("Keyboard interrupt detected. Terminating process...")

    finally:
        if process.poll() is None:
            process.terminate()

        if not STREAM_CHILD_OUTPUT:
            try:
                stdout, stderr = process.communicate(timeout=5)
                logger.info(f"Subprocess output: {stdout.decode(errors='ignore')}")
                logger.error(f"Subprocess errors: {stderr.decode(errors='ignore')}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.error("Subprocess timed out and was killed.")

        # with monitor.lock:
        #     pose = monitor.current_pose
        #     angles = monitor.final_angles

        if landing_pose:
            final_positions.append((landing_pose.x, landing_pose.y, landing_pose.z))
            final_euler_angles.append(landing_angles)
            logger.info(f"Final position recorded: x={landing_pose.x:.3f}, y={landing_pose.y:.3f}, z={landing_pose.z:.3f}")
        else:
            final_positions.append((np.nan, np.nan, np.nan))
            final_euler_angles.append((np.nan, np.nan, np.nan))
            logger.warning("No final position recorded; appending NaN values.")

    if stop_reason is None:
        stop_reason = f"Exited (returncode={process.returncode})"
    return stop_reason, landing_pose     # ← CHANGED: return tuple



def generate_stratified_z(z_min, z_max, n_strata, samples_per_stratum, shuffle=True):
    """
    Generate stratified random z samples.

    Divides [z_min, z_max] into n_strata equal-width bands and draws
    samples_per_stratum uniform random samples from each band.

    Args:
        z_min:               Lower bound of z range.
        z_max:               Upper bound of z range.
        n_strata:            Number of equal-width strata.
        samples_per_stratum: Number of samples to draw from each stratum.
        shuffle:             If True, randomize run order to avoid ordering bias.

    Returns:
        List of z values, length = n_strata * samples_per_stratum.
    """
    band_width = (z_max - z_min) / n_strata
    samples = []

    for s in range(n_strata):
        band_low  = z_min + s * band_width
        band_high = z_min + (s + 1) * band_width
        for _ in range(samples_per_stratum):
            samples.append(np.random.uniform(band_low, band_high))

        logger.info(
            f"Stratum {s + 1}: z in [{band_low:.1f}, {band_high:.1f}) — "
            f"{samples_per_stratum} sample(s) drawn"
        )

    if shuffle:
        np.random.shuffle(samples)
        logger.info("Stratified z samples shuffled to remove ordering bias.")

    return samples




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations with adjustable timeout.")
    parser.add_argument("--timeout", type=int, default=150, help="Timeout duration in seconds.")
    args = parser.parse_args()

    rospy.init_node("simulation_monitor", anonymous=True)
    monitor = SimulationMonitor()
    N_STRATA = 3
    SAMPLES_PER_STRATUM = 5

    stratified_z = generate_stratified_z(
    z_min=z_min, z_max=z_max,
    n_strata=N_STRATA,
    samples_per_stratum=SAMPLES_PER_STRATUM,
    shuffle=True,
    )
    n_trials = len(stratified_z)


    try:
        max_xy_offset_cap = 50
        rate = rospy.Rate(10)

        for i, init_z in enumerate(stratified_z):
            z_offset = abs(init_z - ideal_z)
            max_xy_offset = min(z_offset, max_xy_offset_cap)

            x_range = (ideal_x - max_xy_offset, ideal_x + max_xy_offset)
            y_range = (ideal_y - max_xy_offset, ideal_y + max_xy_offset)

            init_x = np.random.uniform(max(x_min, x_range[0]), min(x_max, x_range[1]))
            init_y = np.random.uniform(max(y_min, y_range[0]), min(y_max, y_range[1]))

            initial_positions.append((init_x, init_y, init_z))

            logger.info(f"Starting iteration {i + 1} with init: x={init_x:.2f}, y={init_y:.2f}, z={init_z:.2f}")
            t0 = time.time()
            stop_reason, landing_pose= run_simulation(monitor, init_x, init_y, init_z, args.timeout)
            landing_times.append(time.time() - t0)
            logger.info(f"Iteration {i + 1} stopped. Reason: {stop_reason}")

            success = (
                landing_pose is not None
                and abs(landing_pose.z - ideal_z) < 2
                and abs(landing_pose.x - ideal_x) <= 4
                and abs(landing_pose.y - ideal_y) <= 4
            )
            landing_results.append("Success" if success else "Fail")

            rate.sleep()

    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        os._exit(0)

    run_folder = create_run_folder()
    save_results_to_csv(run_folder, initial_positions, final_positions, landing_times, landing_results, final_euler_angles)
    print(f"Plots saved in {run_folder}")

    save_summary_to_csv_and_metadata(base_dir="runs")
    print("Performance summary saved.")

    with open(constants.simulation_status_file, "w") as status_file:
        status_file.write("True")
    logger.info("Simulation status updated to True.")
