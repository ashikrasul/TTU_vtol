import os
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define ideal positions
# -48, 134, 6
IDEAL_X = -48
IDEAL_Y = 134
IDEAL_Z = 7

x = [-48]
y = [134]

font_size = 9
fig_width = 3.25
fig_height = 3.25

# Function to plot initial positions
def plot_initial_pos(run_folder, initial_positions, ideal_x, ideal_y):
    """Plot and save initial positions."""
    initial_x, initial_y, initial_z = zip(*initial_positions)
    size_based_on_z = 3000 / (1 + abs(np.array(initial_z) - 35))

    plt.figure(figsize=(fig_width, fig_height))
    scatter = plt.scatter(
        initial_x, initial_y,
        c=abs(np.array(initial_z) - IDEAL_Z),
        cmap='Blues',
        # s=size_based_on_z,
        s=10,
        edgecolor='black',
        linewidth=0.5,
        label='Initial Positions'
    )

    for xi, yi in zip(x, y):
        plt.text(xi, yi, 'H', fontsize=font_size, fontweight='bold', color='red', ha='center', va='center',label='Ideal Landing Position')


    # plt.scatter(ideal_x, ideal_y, color='red', s=30, marker='H', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    cbar = plt.colorbar(scatter, label='Height from Landing Pad (m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad (m)', fontsize=font_size)  # Label font size
    cbar.ax.tick_params(labelsize=9)  # Tick font size

    ax = plt.gca()
    # rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, linewidth=1.5, edgecolor='red',linestyle='dotted', facecolor='none')
    # ax.add_patch(rect)

    # 3. Filled box with transparency
    filled_rect = Rectangle((ideal_x-4, ideal_y-4), 8, 8, 
                        linewidth=0, edgecolor='none', facecolor='red', alpha=0.25)
    ax.add_patch(filled_rect)




    plt.title("Initial QuadPlane Positions", fontsize=font_size,  color='black')  # Title styling
    plt.xlabel("X Position (m)", fontsize=font_size, color='black')   # X-axis label styling
    plt.ylabel("Y Position (m)", fontsize=font_size, color='black')   # Y-axis label styling
    plt.xticks(fontsize=font_size)  # Set the font size of x-axis ticks
    plt.yticks(fontsize=font_size)  # Set the font size of y-axis ticks

    # plt.legend(
    #     loc='upper center', 
    #     bbox_to_anchor=(0.5, -0.1), 
    #     ncol=2, 
    #     borderaxespad=0., 
    #     fontsize=font_size,   # Font size for legend text
    #     edgecolor='black',  # Border color of legend
    #     facecolor='lightgray'  # Background color of legend
    # )

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)  # Grid line style
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(f"{run_folder}/initial_positions.PNG",dpi=300)  # Save as PDF
    plt.close()

# Function to plot final positions
def plot_final_pos(run_folder, final_positions, ideal_x, ideal_y):
    """Plot and save final positions."""
    final_x, final_y, final_z = zip(*final_positions)
    plt.figure(figsize=(fig_width, fig_height))

    for xi, yi in zip(x, y):
        plt.text(xi, yi, 'H', fontsize=font_size, color='red',fontweight='bold', ha='center', va='center',zorder=2)



    scatter_final = plt.scatter(
        final_x, final_y,
        c=abs(np.array(final_z) - IDEAL_Z),
        cmap='Greens',
        s=10,
        edgecolor='black',
        linewidth=0.5,
        label='Final Positions',
        vmin=0, vmax=10, zorder=3
    )





    # plt.scatter(ideal_x, ideal_y, color='red', s=30, marker='H', edgecolor='black', linewidth=0.5, label='Ideal Landing Position')
    cbar = plt.colorbar(scatter_final, label='Height from Landing Pad (m)', orientation='vertical')
    cbar.ax.set_ylabel('Height from Landing Pad (m)', fontsize=font_size)  # Label font size
    cbar.ax.tick_params(labelsize=9)  # Tick font size

    filled_rect = Rectangle((ideal_x-4, ideal_y-4), 8, 8, 
        linewidth=0, edgecolor='none', facecolor='red', alpha=0.25, zorder=1)
    
    ax = plt.gca()
    # rect = Rectangle((ideal_x - 2, ideal_y - 2), 4, 4, linewidth=1.5, edgecolor='red',linestyle='dotted', facecolor='none')
    # ax.add_patch(rect)




    ax.add_patch(filled_rect)


    plt.title("Final Landing Positions", fontsize=font_size, color='black')  # Title styling
    plt.xlabel("X Position (m)", fontsize=font_size, color='black')   # X-axis label styling
    plt.ylabel("Y Position (m)", fontsize=font_size, color='black')   # Y-axis label styling
    plt.xticks(fontsize=font_size)  # Set the font size of x-axis ticks
    plt.yticks(fontsize=font_size)  # Set the font size of y-axis ticks

    # plt.xticks(ticks=range(int(ideal_x - 50), int(ideal_y + 50), 10),fontsize=font_size)  # Set the font size of x-axis ticks
    # plt.yticks(ticks=range(int(ideal_y - 50), int(ideal_y + 50), 10),fontsize=font_size)  # Set the font size of y-axis ticks


    # plt.legend(
    #     loc='upper center', 
    #     bbox_to_anchor=(0.5, -0.5), 
    #     ncol=2, 
    #     borderaxespad=0., 
    #     fontsize=font_size,   # Font size for legend text
    #     edgecolor='black',  # Border color of legend
    #     facecolor='lightgray'  # Background color of legend
    # )


    

    plt.grid(True, color='black', linestyle='--', linewidth=0.5)  # Grid line style
    plt.xlim(ideal_x - 50, ideal_x + 50)
    plt.ylim(ideal_y - 50, ideal_y + 50)
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(f"{run_folder}/final_positions.PNG", dpi=300)  # Save as PDF
    plt.close()

# Main script
def main():
    # Input CSV file path
    input_csv = "./vehicles/jaxguam/runs/run_204/simulation_results.csv"  # Update with your CSV file path
    output_folder = "./plots/laplace"  # Folder to save plots
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
