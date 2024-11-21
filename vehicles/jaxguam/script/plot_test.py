import csv
import os
import re
import sys

from loguru import logger
import numpy as np

sys.path.append(os.path.abspath("/home/arasul42@tntech.edu/works/rraaa-sim/utils"))


from utils import constants



def save_summary_to_csv_and_metadata(base_dir='/home/arasul42@tntech.edu/works/rraaa-sim/vehicles/jaxguam/runs', summary_file="performance_summary.csv"):
    """
    Save the latest run's performance summary and append additional data to the existing metadata in .txt format.
    """
    # Paths for metadata and summary
    metadata_path = constants.metadata_file_path
    summary_path = os.path.join(os.path.dirname(metadata_path), summary_file)

    # Ensure the metadata directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Retrieve current run's folder
    runs = [
        folder for folder in os.listdir(base_dir)
        if folder.startswith("run_") and os.path.isdir(os.path.join(base_dir, folder))
    ]

    if not runs:
        logger.warning("No run folders found.")
        return

    latest_run = sorted(runs, key=lambda x: int(re.search(r'\d+', x).group()))[0]
    
    logger.info(f"Showing data for {latest_run}")  # Get the most recent run folder
    run_path = os.path.join(base_dir, latest_run)
    results_file = os.path.join(run_path, "simulation_results.csv")

    if not os.path.exists(results_file):
        logger.warning(f"Results file not found for {latest_run}. Skipping.")
        return

    # Process results for the latest run
    with open(results_file, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Parse data from simulation results
    final_positions = []
    euler_angles = []
    for row in rows:
        try:
            pos = [float(row['x_final']), float(row['y_final']), float(row['z_final'])]
            angles = [float(row['final_roll']), float(row['final_pitch']), float(row['final_yaw'])]
            final_positions.append(pos)
            euler_angles.append(angles)
        except ValueError:
            logger.warning(f"Skipping row with missing or invalid data: {row}")

    if not final_positions:
        logger.warning(f"No valid final positions for {latest_run}. Skipping run.")
        return

    # Compute deviations and metrics
    ideal_position = [constants.ideal_x, constants.ideal_y, constants.ideal_z]
    deviations = [
        ((p[0] - ideal_position[0]) ** 2 + (p[1] - ideal_position[1]) ** 2 + (p[2] - ideal_position[2]) ** 2) ** 0.5
        for p in final_positions
    ]
    std_dev = np.std(deviations)
    success_rate = sum(1 for row in rows if row['Landing Result (Success/Fail)'] == 'Success') / len(rows)

    # Read existing metadata from the .txt file
    existing_metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as meta_file:
            for line in meta_file:
                key, value = line.strip().split(':', 1)
                existing_metadata[key.strip()] = value.strip()

    # Append additional data to the metadata
    existing_metadata.update({
        "run_number": latest_run,
        "std_dev": f"{std_dev:.6f}",
        "success_rate": f"{success_rate:.6f}",
        "final_positions": str(final_positions),
        "euler_angles": str(euler_angles)
    })

    # Save updated metadata back to the .txt file
    with open(metadata_path, 'w') as meta_file:
        for key, value in existing_metadata.items():
            meta_file.write(f"{key}: {value}\n")

    # Append the latest run's summary to the summary CSV
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_header = [
        "Run Number", "Std Dev (Position)", "Success Rate", "Model Name", "Scale", "HSV_V"
    ]
    run_summary = [
        latest_run, std_dev, success_rate,
        existing_metadata.get("model_name", "N/A"),
        existing_metadata.get("scale", "N/A"),
        existing_metadata.get("hsv_v", "N/A")
    ]

    if not os.path.exists(summary_path):
        # Create new CSV file with header
        with open(summary_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summary_header)
            writer.writerow(run_summary)
    else:
        # Append to existing CSV
        with open(summary_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(run_summary)

    logger.info(f"Performance summary updated for run {latest_run} in {summary_path}")
    logger.info(f"Metadata updated with the latest run: {metadata_path}")

if __name__ == "__main__":
    save_summary_to_csv_and_metadata()