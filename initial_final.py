import os
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define ideal positions
IDEAL_X = -80
IDEAL_Y = 75
IDEAL_Z = 35

x = [-80]
y = [75]

# Function to plot initial positions
def plot_initial_pos(run_folder, initial_positions, ideal_x, ideal_y):
    """Plot and save initial positions."""
    initial_x, initial_y, initial_z = zip(*initial_positions)
    size_based_on_z = 3000 / (1 + abs(np.array(initial_z) - 35))

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        initial_x, initial_y,
        c=abs(np.array(initial_z) - 35),
        cmap='Blues_r',
        s=size_based_on_z,
        edgecolor='black',
        linewidth=0.5,
        label='Initial Positions'
    )

    for xi, yi in zip(x, y):
        plt.text(xi, yi, 'H', fontsize=14, color='red', ha='center', va='center',label='Ideal Landing Position')


    # plt.scatter(ideal_x, ideal_y, color='red', s=30, marker='H', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    cbar = plt.colorbar(scatter, label='Height from Landing Pad (m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad(m)', fontsize=18)  # Label font size
    cbar.ax.tick_params(labelsize=18)  # Tick font size

    ax = plt.gca()
    # rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, linewidth=1.5, edgecolor='red',linestyle='dotted', facecolor='none')
    # ax.add_patch(rect)

    # 3. Filled box with transparency
    filled_rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, 
                        linewidth=0, edgecolor='none', facecolor='red', alpha=0.25)
    ax.add_patch(filled_rect)




    plt.title("Initial VTOL Positions", fontsize=20, fontweight='bold', color='black')  # Title styling
    plt.xlabel("X Position (m)", fontsize=18, fontweight='bold', color='black')   # X-axis label styling
    plt.ylabel("Y Position (m)", fontsize=18, fontweight='bold', color='black')   # Y-axis label styling
    plt.xticks(fontsize=18)  # Set the font size of x-axis ticks
    plt.yticks(fontsize=18)  # Set the font size of y-axis ticks

    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=2, 
        borderaxespad=0., 
        fontsize=18,   # Font size for legend text
        edgecolor='black',  # Border color of legend
        facecolor='lightgray'  # Background color of legend
    )

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)  # Grid line style
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(f"{run_folder}/initial_positions.pdf")  # Save as PDF
    plt.close()

# Function to plot final positions
def plot_final_pos(run_folder, final_positions, ideal_x, ideal_y):
    """Plot and save final positions."""
    final_x, final_y, final_z = zip(*final_positions)
    plt.figure(figsize=(10, 8))



    for xi, yi in zip(x, y):
        plt.text(xi, yi, 'H', fontsize=14, color='red', ha='center', va='center')



    scatter_final = plt.scatter(
        final_x, final_y,
        c=abs(np.array(final_z) - 35),
        cmap='Greens',
        s=100,
        edgecolor='black',
        linewidth=0.5,
        label='Final Positions'
    )





    # plt.scatter(ideal_x, ideal_y, color='red', s=30, marker='H', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    cbar = plt.colorbar(scatter_final, label='Height from Landing Pad(m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad(m)', fontsize=18)  # Label font size
    cbar.ax.tick_params(labelsize=18)  # Tick font size


    ax = plt.gca()
    # rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, linewidth=1.5, edgecolor='red',linestyle='dotted', facecolor='none')
    # ax.add_patch(rect)



    filled_rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, 
        linewidth=0, edgecolor='none', facecolor='red', alpha=0.25)
    ax.add_patch(filled_rect)


    plt.title("Final Landing Positions", fontsize=20, fontweight='bold', color='black')  # Title styling
    plt.xlabel("X Position (m)", fontsize=18, fontweight='bold', color='black')   # X-axis label styling
    plt.ylabel("Y Position (m)", fontsize=18, fontweight='bold', color='black')   # Y-axis label styling
    plt.xticks(fontsize=18)  # Set the font size of x-axis ticks
    plt.yticks(fontsize=18)  # Set the font size of y-axis ticks


    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=2, 
        borderaxespad=0., 
        fontsize=18,   # Font size for legend text
        edgecolor='black',  # Border color of legend
        facecolor='lightgray'  # Background color of legend
    )


    

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)  # Grid line style
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(f"{run_folder}/final_positions.pdf")  # Save as PDF
    plt.close()

# Main script
def main():
    # Input CSV file path
    input_csv = "/home/arasul42@tntech.edu/works/rraaa-sim/vehicles/jaxguam/runs/run_106/simulation_results.csv"  # Update with your CSV file path
    output_folder = "/home/arasul42@tntech.edu/works/rraaa-sim/plots"  # Folder to save plots
    os.makedirs(output_folder, exist_ok=True)

    # Read data from CSV
    df = pd.read_csv(input_csv)

    # Extract initial positions
    initial_positions = list(zip(df['x_i'], df['y_i'], df['z_i']))

    # Extract final positions
    final_positions = list(zip(df['x_final'], df['y_final'], df['z_final']))

    # Generate plots
    print("Generating plots...")
    plot_initial_pos(output_folder, initial_positions, IDEAL_X, IDEAL_Y)
    plot_final_pos(output_folder, final_positions, IDEAL_X, IDEAL_Y)
    print(f"Plots saved in {output_folder}")

if __name__ == "__main__":
    main()
