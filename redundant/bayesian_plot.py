from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import cdist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_process_plots_centered_on_data(csv_path, optimization_run):
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Filter data by optimization run
    df = df[df["Optimization Run"].isin(optimization_runs)]

    print(df)

    # Ensure data exists for the selected run
    if df.empty:
        print(f"No data found for optimization run {optimization_run}.")
        return

    # Prepare input data
    X = df[["Scale", "HSV_V"]].values
    y = df["Success Rate"].values

    # Define Gaussian Process kernel function (RBF kernel)
    def rbf_kernel(X1, X2, length_scale=0.06, sigma_f=1.0):
        dists = cdist(X1, X2, metric='euclidean')
        return sigma_f**2 * np.exp(-0.5 * (dists / length_scale) ** 2)

    # Kernel matrices for the given data
    K = rbf_kernel(X, X, length_scale=0.005)  # Use a smaller length_scale for tighter contours
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))  # Add small noise for numerical stability

    # Create a grid for predictions
    x_grid = np.linspace(0, 1, 100)
    y_grid = np.linspace(0, 1, 100)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    X_pred = np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T

    # Compute the mean predictions for the Gaussian Process
    K_s = rbf_kernel(X_pred, X, length_scale=0.05)  # Match the kernel length_scale
    mean_pred = K_s @ K_inv @ y

    # Reshape predictions for plotting
    z_pred = mean_pred.reshape(x_mesh.shape)

    # Plot the Gaussian Process
    plt.figure(figsize=(12, 10))
    contour = plt.contourf(x_mesh, y_mesh, z_pred, levels=50, cmap="inferno", alpha=0.9, vmin=0, vmax=1)
    cbar=plt.colorbar(contour, label="Predicted Success Rate")
    cbar.ax.set_ylabel("Predicted Success Rate", fontsize=18,fontweight='bold', color='black')  # Label font size
    cbar.ax.tick_params(labelsize=18)  # Tick font size

    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))







    # Overlay data points
    plt.scatter(df["Scale"], df["HSV_V"], color="black", marker="o", s=10, label="Data Points")



    # Add labels and legend
    plt.xlabel("Scale",fontsize=20, fontweight='bold', color='black')
    plt.ylabel("Brightness Value (hsv_v)",fontsize=20, fontweight='bold', color='black')
    # plt.title(f"Success Rate Evolution with Training", fontsize=22, fontweight='bold', color='black')  # Title styling
    plt.xticks(fontsize=20)  # Set the font size of x-axis ticks
    plt.yticks(fontsize=20)  # Set the font size of y-axis ticks

    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.1), 
        ncol=2, 
        borderaxespad=0., 
        fontsize=20,   # Font size for legend text
        edgecolor='black',  # Border color of legend
        facecolor='lightgray'  # Background color of legend
    )

    plt.tight_layout()
    plt.show()

# Example usage:
# Provide the CSV file path and the optimization run number
csv_path = "./utils/performance_summary.csv"  # Replace with your CSV file path
optimization_runs = [10,12] # Replace with the desired optimization run number
generate_gaussian_process_plots_centered_on_data(csv_path, optimization_runs)
