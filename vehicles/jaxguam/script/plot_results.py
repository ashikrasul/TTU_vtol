import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from loguru import logger

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