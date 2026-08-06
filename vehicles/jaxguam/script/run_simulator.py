#!/usr/local/bin/py310ros
import os
import re
import sys
import time
import yaml
import rospy
import numpy as np
import subprocess
import argparse
from threading import Lock
from collections import deque
from geometry_msgs.msg import PoseStamped, Vector3
from tf.transformations import euler_from_quaternion
from utils import constants
from loguru import logger
from plot_results import (
    create_run_folder,
    save_results_to_csv,
    save_summary_to_csv_and_metadata,
)

from utils.config import load_yaml_file, write_shared_tmp_file, update_metadata


config_path = constants.merged_config_path
temp_config_path = "/tmp/tmp_config.yml"

config = load_yaml_file(config_path)

ue_variant      = 'ue5' if config.get('carla_key', 'carla_ue4') == 'carla_ue5' else 'ue4'
ideal_x         = config['ideal_position'][ue_variant]['x']
ideal_y         = config['ideal_position'][ue_variant]['y']
ideal_z         = config['ideal_position'][ue_variant]['z']

TOL_X           = config['landing_tolerance']['x']
TOL_Y           = config['landing_tolerance']['y']
TOL_Z           = config['landing_tolerance']['z']

EPISODE_TIMEOUT     = config['episode_timeout']
ZERO_VEL_STOP_SEC   = config['zero_vel_stop_sec']
STUCK_WINDOW_SEC     = config['stuck_window_sec']
STUCK_DISPLACEMENT_M = config['stuck_displacement_m']
STUCK_MIN_ALTITUDE_M = config['stuck_min_altitude_m']

range_offset    = [config['range_offset']['x'], config['range_offset']['y'], config['range_offset']['z']]
x_min, x_max    = ideal_x - range_offset[0], ideal_x + range_offset[0]
y_min, y_max    = ideal_y - range_offset[1], ideal_y + range_offset[1]
z_min, z_max    = ideal_z + config['z_min_offset'], ideal_z + range_offset[2]

INIT_WAIT_TIMEOUT   = config['init_gating']['wait_timeout']
FRESH_MSG_MAX_AGE   = config['init_gating']['fresh_msg_max_age']
INIT_TOL_XY         = config['init_gating']['tol_xy']
INIT_TOL_Z          = config['init_gating']['tol_z']
MAX_INIT_RETRIES    = config['init_gating'].get('max_retries', 2)

N_STRATA            = config['stratification']['n_strata']
SAMPLES_PER_STRATUM = config['stratification']['samples_per_stratum']

initial_positions = []
final_positions = []
landing_times = []
landing_results = []
final_euler_angles = []
stop_reasons = []

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
            if self.episode_started and msg.pose.position.z <= ideal_z:
                self.z_value_below_threshold = True
                q = (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                )
                self.final_angles = euler_from_quaternion(q)

            if self.episode_started:
                p = msg.pose.position

                # Track recent positions for the stuck/no-progress check
                # (is_stalled): keep only the last STUCK_WINDOW_SEC seconds.
                now = time.time()
                self.position_history.append((now, p.x, p.y, p.z))
                while self.position_history and (now - self.position_history[0][0]) > STUCK_WINDOW_SEC:
                    self.position_history.popleft()

                # "Within tolerance" once x/y are within their tolerances and
                # z has descended to ideal_z (not a +/- band).
                if (abs(p.x - ideal_x) <= TOL_X and
                    abs(p.y - ideal_y) <= TOL_Y and
                    p.z <= ideal_z):
                    self.within_tolerance = True
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
            self.within_tolerance = False

            self.velocity_zero_start_time = None
            self.zero_velocity_duration = 0.0

            self.position_history = deque()

    def is_stalled(self):
        """True if, above STUCK_MIN_ALTITUDE_M, the net 3D displacement over
        the last STUCK_WINDOW_SEC seconds is below STUCK_DISPLACEMENT_M —
        i.e. the UAV is oscillating without making progress."""
        with self.lock:
            if self.current_pose is None or self.current_pose.z <= STUCK_MIN_ALTITUDE_M:
                return False
            if not self.position_history:
                return False
            oldest_t, oldest_x, oldest_y, oldest_z = self.position_history[0]
            if (time.time() - oldest_t) < STUCK_WINDOW_SEC:
                return False
            p = self.current_pose
            displacement = ((p.x - oldest_x) ** 2 + (p.y - oldest_y) ** 2 + (p.z - oldest_z) ** 2) ** 0.5
            return displacement < STUCK_DISPLACEMENT_M


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


def run_simulation(monitor, init_x, init_y, init_z, timeout=EPISODE_TIMEOUT):
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
                in_tol = monitor.within_tolerance

            if in_tol:
                with monitor.lock:
                    landing_pose = monitor.current_pose
                    landing_angles = monitor.final_angles
                stop_reason = "Within tolerance"
                logger.info(f"Iteration stopped: {stop_reason}")
                break

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

            if monitor.is_stalled():
                with monitor.lock:
                    landing_pose = monitor.current_pose
                    landing_angles = monitor.final_angles
                stop_reason = f"No progress (<{STUCK_DISPLACEMENT_M}m) over {STUCK_WINDOW_SEC}s"
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

        # Fall back to last known pose if stop condition fired but pose was None,
        # or if subprocess crashed before any stop condition.
        if landing_pose is None:
            with monitor.lock:
                if monitor.current_pose is not None:
                    landing_pose = monitor.current_pose
                    landing_angles = monitor.final_angles
                    logger.warning("landing_pose was None at stop; using last known pose.")

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
    parser.add_argument("--timeout", type=int, default=EPISODE_TIMEOUT, help="Timeout duration in seconds.")
    args = parser.parse_args()

    rospy.init_node("simulation_monitor", anonymous=True)
    monitor = SimulationMonitor()

    # ── Pre-flight readiness gate ──────────────────────────────────────────
    # Confirms the full stack (CARLA → env_sim → jaxguam) is actually
    # publishing pose BEFORE we burn n_trials evaluation episodes on a dead
    # simulator. Without this, a "CARLA not ready" condition silently
    # produces 15 timed-out/NaN episodes → a fabricated 0% success rate that
    # gets logged to performance_summary.csv indistinguishable from a real
    # 0% landing-model result, corrupting the BO surrogate.
    READY_TIMEOUT = INIT_WAIT_TIMEOUT * 3
    rate = rospy.Rate(10)
    deadline = time.time() + READY_TIMEOUT
    sim_ready = False
    while time.time() < deadline and not rospy.is_shutdown():
        with monitor.lock:
            if monitor.current_pose is not None and _pose_is_fresh(monitor):
                sim_ready = True
                break
        rate.sleep()

    if not sim_ready:
        logger.error(
            f"No fresh /jaxguam/pose received within {READY_TIMEOUT}s — "
            f"CARLA/simulation stack is not ready. Aborting WITHOUT running "
            f"trials or writing performance results (would otherwise record "
            f"a fabricated 0% success rate)."
        )
        update_metadata(
            fields={"sim_ready": "False"},
            meta_file_path=constants.metadata_file_path,
        )
        # Unblock rraaa.py's polling loop so containers still get torn down.
        with open(constants.simulation_status_file, "w") as status_file:
            status_file.write("True")
        sys.exit(1)

    update_metadata(
        fields={"sim_ready": "True"},
        meta_file_path=constants.metadata_file_path,
    )
    logger.info(f"Simulation stack ready (pose stream confirmed). Proceeding with {N_STRATA * SAMPLES_PER_STRATUM} trials.")

    # ── Seed RNG deterministically from (BO seed, iteration number) ────────
    # Without this, generate_stratified_z() and the per-trial init-position
    # sampling below draw from the unseeded global numpy RNG, so repeated
    # runs get a different set of 15 landing scenarios each time. Combining
    # the fixed BO seed with the per-iteration run number (from
    # metadata['run_number'], e.g. "run_341" -> 341) makes the scenario set
    # reproducible across reruns of the same BO trajectory while still
    # varying from one BO query to the next — independent of (scale, hsv_v).
    base_seed = config.get('bayesian_optimisation', {}).get('seed')
    meta = load_yaml_file(constants.metadata_file_path) if os.path.exists(constants.metadata_file_path) else {}
    run_number_str = meta.get('run_number')
    run_num_match = re.search(r'\d+', str(run_number_str)) if run_number_str else None
    if base_seed is not None and run_num_match is not None:
        run_num = int(run_num_match.group())
        sim_seed = (int(base_seed) + run_num) % (2 ** 32)
        np.random.seed(sim_seed)
        logger.info(f"RNG seeded with {sim_seed} (base_seed={base_seed}, run_num={run_num}).")
    else:
        logger.warning("No seed/run_number found in config/metadata — RNG not seeded (non-reproducible scenarios).")

    stratified_z = generate_stratified_z(
    z_min=z_min, z_max=z_max,
    n_strata=N_STRATA,
    samples_per_stratum=SAMPLES_PER_STRATUM,
    shuffle=True,
    )
    n_trials = len(stratified_z)


    try:
        # Altitude-dependent xy spawn offset: scales with height above the
        # target (init_z - ideal_z), so higher spawns get a wider xy spread
        # than lower spawns — inspired by the RANGE_BY_ALT approach in
        # rraaa-sim's POMCP_CARLA/run_fov_experiment.py. At FOV=90° the
        # ground footprint half-width equals the height above target, so
        # ALT_XY_RATIO < 1 keeps the helipad within the camera FOV at every
        # altitude. Final result is still clamped to [x_min, x_max] /
        # [y_min, y_max] (range_offset.x/y) below.
        ALT_XY_RATIO = 0.6
        rate = rospy.Rate(10)

        for i, init_z in enumerate(stratified_z):
            max_xy_offset = ALT_XY_RATIO * (init_z - ideal_z)

            x_range = (ideal_x - max_xy_offset, ideal_x + max_xy_offset)
            y_range = (ideal_y - max_xy_offset, ideal_y + max_xy_offset)

            init_x = np.random.uniform(max(x_min, x_range[0]), min(x_max, x_range[1]))
            init_y = np.random.uniform(max(y_min, y_range[0]), min(y_max, y_range[1]))

            initial_positions.append((init_x, init_y, init_z))

            logger.info(f"Starting iteration {i + 1} with init: x={init_x:.2f}, y={init_y:.2f}, z={init_z:.2f}")
            t0 = time.time()
            for attempt in range(1, MAX_INIT_RETRIES + 2):
                stop_reason, landing_pose = run_simulation(monitor, init_x, init_y, init_z, args.timeout)
                if not stop_reason.startswith("Init failed"):
                    break
                # Sim infrastructure hiccup (vehicle never reached its spawn
                # point) — not a model-performance result. Discard the
                # bookkeeping this attempt appended and retry the same
                # (init_x, init_y, init_z) with a fresh land_vtol.py process.
                final_positions.pop()
                final_euler_angles.pop()
                logger.warning(f"Iteration {i + 1}: init failed (attempt {attempt}/{MAX_INIT_RETRIES + 1}) — retrying...")
            landing_times.append(time.time() - t0)
            logger.info(f"Iteration {i + 1} stopped. Reason: {stop_reason}")

            success = (
                landing_pose is not None
                and abs(landing_pose.z - ideal_z) < TOL_Z
                and abs(landing_pose.x - ideal_x) <= TOL_X
                and abs(landing_pose.y - ideal_y) <= TOL_Y
            )
            landing_results.append("Success" if success else "Fail")
            stop_reasons.append(stop_reason)

            rate.sleep()

    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        os._exit(0)

    run_folder = create_run_folder()
    save_results_to_csv(run_folder, initial_positions, final_positions, landing_times, landing_results, final_euler_angles, stop_reasons)
    print(f"Plots saved in {run_folder}")

    save_summary_to_csv_and_metadata(base_dir="runs")
    print("Performance summary saved.")

    with open(constants.simulation_status_file, "w") as status_file:
        status_file.write("True")
    logger.info("Simulation status updated to True.")
