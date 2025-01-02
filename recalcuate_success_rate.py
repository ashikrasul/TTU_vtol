import os
import pandas as pd
import numpy as np

# Define the directories and ideal pose
base_dir = '/home/arasul42@tntech.edu/works/rraaa-sim/vehicles/jaxguam/runs'
run_folders = [f"run_{i}" for i in range(127, 180)]
output_csv = "summary_results.csv"
ideal_x, ideal_y, ideal_z = -80, 75, 35

# Initialize results list
summary_results = []

# Iterate through each run folder
for folder in run_folders:
    folder_path = os.path.join(base_dir, folder)
    csv_path = os.path.join(folder_path, "simulation_results.csv")
    
    if os.path.exists(csv_path):
        # Read the CSV file
        data = pd.read_csv(csv_path)

        # Calculate deviations and success
        deviations = []
        success_count = 0
        total_count = len(data)
        
        for _, row in data.iterrows():
            x_final, y_final, z_final = row["x_final"], row["y_final"], row["z_final"]
            deviation = np.sqrt(
                (x_final - ideal_x) ** 2 +
                (y_final - ideal_y) ** 2 +
                (z_final - ideal_z) ** 2
            )
            deviations.append(deviation)
            
            # Check success condition
            is_success = (
                abs(z_final - ideal_z) < 1 and
                abs(x_final - ideal_x) <= 4 and
                abs(y_final - ideal_y) <= 4
            )
            if is_success:
                success_count += 1
        
        # Calculate standard deviation and success rate
        std_dev_position = np.std(deviations)
        success_rate = (success_count / total_count)if total_count > 0 else 0
        
        # Store summary data
        summary_results.append({
            "Run Number": folder.split("_")[1],  # Extract run number
            "Std Dev (Position)": std_dev_position,
            "Success Rate": success_rate
        })

# Save results to a new CSV file
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(output_csv, index=False)

# Print confirmation
print(f"Summary results saved to {output_csv}")
