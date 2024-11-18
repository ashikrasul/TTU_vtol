import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from loguru import logger



ideal_x, ideal_y, ideal_z = -80, 75, 32.7

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


def save_results_to_csv(run_folder, initial_positions, final_positions, landing_times, landing_results, final_euler_angles):
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
            "final_roll", "final_pitch", "final_yaw"
        ])
        
        for i in range(len(initial_positions)):
            roll, pitch, yaw = (final_euler_angles[i] if final_euler_angles[i] else (None, None, None))
            writer.writerow([
                initial_positions[i][0], 
                initial_positions[i][1], 
                initial_positions[i][2],
                landing_times[i],
                landing_results[i],
                final_positions[i][0],
                final_positions[i][1],
                final_positions[i][2],
                roll, pitch, yaw
            ])
    logger.info(f"Results saved to {csv_file_path}")

def save_summary_to_csv(base_dir="runs", summary_file="performance_summary.csv"):
    """Save performance summary of all runs in a single CSV file."""
    summary_path = os.path.join(base_dir, summary_file)
    
    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist.")
        return
    
    runs = [folder for folder in os.listdir(base_dir) if folder.startswith("run_") and os.path.isdir(os.path.join(base_dir, folder))]
    if not runs:
        logger.warning("No run folders found.")
        return
    
    summary_data = []
    
    for run in runs:
        run_path = os.path.join(base_dir, run)
        results_file = os.path.join(run_path, "simulation_results.csv")
        
        if not os.path.exists(results_file):
            logger.warning(f"Results file not found for {run}. Skipping.")
            continue
        
        with open(results_file, 'r') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        
        # Filter valid rows
        final_positions = []
        euler_angles = []
        
        for row in rows:
            try:
                pos = [float(row['x_final']), float(row['y_final']), float(row['z_final'])]
                angles = [float(row['final_roll']), float(row['final_pitch']), float(row['final_yaw'])]
                final_positions.append(pos)
                euler_angles.append(angles)
            except ValueError:
                logger.warning(f"Skipping row with missing or invalid data in run {run}: {row}")
        
        if not final_positions:
            logger.warning(f"No valid final positions for {run}. Skipping run.")
            continue
        
        final_positions = np.array(final_positions)
        euler_angles = np.array(euler_angles)

        ideal_position = np.array([ideal_x, ideal_y, ideal_z])
        deviations = np.linalg.norm(final_positions - ideal_position, axis=1)
        std_dev = np.std(deviations)
        
        max_deviation_position = np.max(np.abs(final_positions - ideal_position), axis=0)
        max_deviation_angles = np.max(np.abs(euler_angles), axis=0)
        
        success_rate = sum(1 for row in rows if row['Landing Result (Success/Fail)'] == 'Success') / len(rows)
        
        summary_data.append([
            run, std_dev, success_rate,
            *max_deviation_position, *max_deviation_angles
        ])
    
    with open(summary_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Run Number", "Std Dev (Position)", "Success Rate",
            "Max Deviation X", "Max Deviation Y", "Max Deviation Z",
            "Max Roll Deviation", "Max Pitch Deviation", "Max Yaw Deviation"
        ])
        writer.writerows(summary_data)
    
    logger.info(f"Performance summary saved to {summary_path}")
