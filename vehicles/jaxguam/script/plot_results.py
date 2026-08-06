import os
import re
import matplotlib.pyplot as plt
import numpy as np
import csv

from loguru import logger

from utils import constants
from utils.config import update_metadata
import yaml




ideal_x, ideal_y, ideal_z = -48, 134, 6

def create_run_folder(base_dir="runs"):
    """Create a unique run folder (run_1, run_2, etc.)."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    run_count = 1
    while os.path.exists(f"{base_dir}/run_{run_count}"):
        run_count += 1
    
    run_folder = f"{base_dir}/run_{run_count}"
    os.makedirs(run_folder)
    os.makedirs(f"{run_folder}/initial_position")
    os.makedirs(f"{run_folder}/landing_position")
    return run_folder

def plot_initial_pos(run_folder, initial_positions, ideal_x, ideal_y):
    """Plot and save initial positions."""
    initial_x, initial_y, initial_z = zip(*initial_positions)
    size_based_on_z = 3000 / (1 + abs(np.array(initial_z) - 35))

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        initial_x, initial_y,
        c=abs(np.array(initial_z) - 35),
        cmap='Blues_r',
        s=size_based_on_z,
        edgecolor='black',
        linewidth=0.5,
        label='Initial Positions'
    )
    plt.scatter(ideal_x, ideal_y, color='red', s=600, marker='o', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    plt.colorbar(scatter, label='Height Difference from Z=35 (m)', orientation='vertical')
    plt.title("Initial Positions with Dynamic Size and Color Scaling")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, borderaxespad=0.)
    plt.grid(True)
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()
    plt.savefig(f"{run_folder}/initial_position/initial_positions.png")
    plt.close()

def plot_final_pos(run_folder, final_positions, ideal_x, ideal_y):
    """Plot and save final positions."""
    final_x, final_y, final_z = zip(*final_positions)

    plt.figure(figsize=(10, 8))
    scatter_final = plt.scatter(
        final_x, final_y,
        c=abs(np.array(final_z) - 35),
        cmap='Greens',
        s=100,
        edgecolor='black',
        linewidth=0.5,
        label='Final Positions'
    )
    plt.scatter(ideal_x, ideal_y, color='red', s=600, marker='o', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    plt.colorbar(scatter_final, label='Height Difference from Z=35 (m)', orientation='vertical')
    plt.title("Final Positions with Color Scaling Only (Fixed Size)")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, borderaxespad=0.)
    plt.grid(True)
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()
    plt.savefig(f"{run_folder}/landing_position/final_positions.png")
    plt.close()


def save_results_to_csv(run_folder, initial_positions, final_positions, landing_times, landing_results, final_euler_angles, stop_reasons=None):
    """Save simulation results to a CSV file."""
    csv_file_path = os.path.join(run_folder, "simulation_results.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow([
            "x_i", "y_i", "z_i",
            "Time to Land (s)",
            "Landing Result (Success/Fail)",
            "x_final", "y_final", "z_final",
            "final_roll", "final_pitch", "final_yaw",
            "Stop Reason"
        ])

        for i in range(len(initial_positions)):
            roll, pitch, yaw = (final_euler_angles[i] if final_euler_angles[i] else (None, None, None))
            stop_reason = stop_reasons[i] if stop_reasons else None
            writer.writerow([
                initial_positions[i][0],
                initial_positions[i][1],
                initial_positions[i][2],
                landing_times[i],
                landing_results[i],
                final_positions[i][0],
                final_positions[i][1],
                final_positions[i][2],
                roll, pitch, yaw,
                stop_reason
            ])
    logger.info(f"Results saved to {csv_file_path}")

def _read_test_model(metadata):
    """Read test_model from node_yolo's dedicated file; fall back to metadata."""
    try:
        with open(constants.test_model_file_path, 'r') as f:
            name = f.read().strip()
            if name:
                return name
    except Exception:
        pass
    return metadata.get("test_model", "N/A")


def save_summary_to_csv_and_metadata(base_dir="runs", summary_file="performance_summary.csv"):
    """
    Save the latest run's performance summary and append additional data to the existing metadata in .txt format.
    """
    # Paths for metadata and summary
    metadata_path = constants.metadata_file_path
    summary_path = os.path.join(os.path.dirname(metadata_path), summary_file)

    # Ensure necessary directories exist
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Find the latest run folder
    runs = sorted(
        [folder for folder in os.listdir(base_dir) if folder.startswith("run_") and os.path.isdir(os.path.join(base_dir, folder))],
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    if not runs:
        logger.warning("No run folders found.")
        return
    latest_run = runs[-1]
    logger.info(f"Processing data for latest run: {latest_run}")

    # Locate results file
    results_file = os.path.join(base_dir, latest_run, "simulation_results.csv")
    if not os.path.exists(results_file):
        logger.warning(f"Results file not found for {latest_run}. Skipping.")
        return

    # Parse results
    try:
        with open(results_file, 'r') as file:
            rows = list(csv.DictReader(file))
    except Exception as e:
        logger.error(f"Error reading results file: {e}")
        return

    # Calculate metrics
    final_positions = [
        [float(row['x_final']), float(row['y_final']), float(row['z_final'])]
        for row in rows if all(key in row for key in ['x_final', 'y_final', 'z_final'])
    ]
    if not final_positions:
        logger.warning(f"No valid final positions for {latest_run}. Skipping run.")
        return

    ideal_position = [constants.ideal_x, constants.ideal_y, constants.ideal_z]
    deviations = [np.linalg.norm(np.array(p) - np.array(ideal_position)) for p in final_positions]
    std_dev = np.std(deviations)
    success_rate = sum(1 for row in rows if row.get('Landing Result (Success/Fail)', '').strip() == 'Success') / len(rows)

    # Read existing metadata
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as meta_file:
            for line in meta_file:
                if ': ' in line:  # Ensure the line contains a key-value pair
                    key, value = line.strip().split(': ', 1)
                    metadata[key.strip()] = value.strip()

    # Fetch Optimization_run value
    optimization_run = metadata.get("optimization_run", "N/A")






    # Update metadata
    metadata.update({
        "run_number": latest_run,
        "std_dev": f"{std_dev:.6f}",
        "success_rate": f"{success_rate:.6f}"
    })

    # Save metadata back to file
    with open(metadata_path, 'w') as file:
        for key, value in metadata.items():
            file.write(f"{key}: {value}\n")
    logger.info(f"Metadata updated: {metadata_path}")

    

    # Append results to CSV
    summary_header = ["Run Number", "Std Dev (Position)", "Success Rate", "Optimization Run", "Model Name", "Training Folder", "Scale", "HSV_V", "Test_model", "opt_success", "acq_fn", "sim_ready"]
    summary_row = [
        latest_run, f"{std_dev:.6f}", f"{success_rate:.6f}", optimization_run,
        metadata.get("model_name", "N/A"),
        metadata.get("training_folder", "N/A"),
        metadata.get("scale", "N/A"),
        metadata.get("hsv_v", "N/A"),
        _read_test_model(metadata),
        metadata.get("opt_success", "N/A"),
        metadata.get("acq_fn", "N/A"),
        metadata.get("sim_ready", "N/A"),
    ]

    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(summary_header)
        writer.writerow(summary_row)

    logger.info(f"Summary CSV updated: {summary_path}")



    # # Load current metadata safely
    # with open(metadata_path, "r") as f:
    #     current_metadata = yaml.safe_load(f) or {}

    # # Create blank version (preserve keys, blank values)
    # blank_metadata = {key: "" for key in current_metadata.keys()}

    # # Update metadata using centralized function
    # update_metadata(
    #     fields=blank_metadata,
    #     meta_file_path=metadata_path
    # )

    # logger.info("All metadata values reset to blank using update_metadata().")


